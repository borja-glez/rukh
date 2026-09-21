"""Publish a trained model to the Hugging Face Hub: weights, ONNX, tokenizer and card.

One repository per stage holds everything somebody needs to reproduce or run the model: the
PyTorch weights, the fp16 and int8 ONNX files the demo loads, the exact vocabulary the model was
trained on, and an English card written from the MLflow run and the evaluation ``results.json``
rather than by hand, so the numbers in the card are the numbers that were measured.

The folder is always staged locally first (under ``artifacts/publish/<repo>/``) and uploaded in
one ``upload_folder`` call; ``dry_run`` stops after staging and touches no network. Every
``HfApi`` call goes through ``_api`` so the tests can replace it, exactly as ``data/publish``
does.

Two kinds of model go out through this one path. A decoder is published with its UCI
vocabulary and the metrics of ``rukh eval``; an encoder (the ``MultiHead`` of M3: the position
encoder plus the value, blunder and result heads) is published with the ``squares`` vocabulary
it reads and the metrics of ``rukh eval encoder``, including the F1 of the material baseline it
has to beat. ``checkpoint_kind`` decides which one a checkpoint is, from the weights themselves,
and the two differ only in the card, the config and the vocabulary file.

The head is tied to the token embedding, so ``lm_head.weight`` and ``tokens.weight`` are one
tensor under two names. ``safetensors`` refuses to serialise that (it stores tensors, not
aliases) and counting both would inflate the parameter count by a whole embedding table, so the
tied name is dropped from what is written and the loader re-ties it: ``MoveDecoder`` builds the
tie in its constructor and ``rukh.train.load_state`` accepts the missing name.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ConfigDict

from rukh import __version__
from rukh.config import BaseConfig
from rukh.data.publish import CARDS_DIR
from rukh.paths import resolve
from rukh.tokenize.uci_vocab import UciTokenizer
from rukh.train.checkpoint import TIED_SOURCES

log = logging.getLogger(__name__)

CARD_TEMPLATE = "model.md.jinja"
ENCODER_CARD_TEMPLATE = "encoder.md.jinja"
PRETRAINED_ENCODER_CARD_TEMPLATE = "encoder-mmm.md.jinja"
"""For a bare ``PositionEncoder``: masked-move pretraining with no heads on it yet."""
CONFIG_NAME = "config.json"
README_NAME = "README.md"
SAFETENSORS_NAME = "model.safetensors"
TORCH_NAME = "pytorch_model.bin"
VOCAB_PATH = "tokenizer/vocab.json"
ONNX_DIR = "onnx"
ONNX_FILES = ("model-fp16.onnx", "model-int8.onnx", "model.onnx")
PARITY_NAME = "parity.json"
"""What ``rukh export`` leaves beside the ONNX files; it is copied and quoted, not recomputed."""
PARITY_BAR = 0.999
"""The parity ``docs/spec/02`` asks of an exported file: the same decision on 99.9 % of them."""
PARITY_DELTAS = ("max_abs_logit_delta", "max_abs_value_delta")
"""The worst-drift key of each kind of parity, in the order the card looks for one."""
PRECISIONS = ("fp32", "fp16", "int8")
"""The order the precisions appear in the card: the reference first, then what is derived."""
PARITY_SOURCES = {
    "validation": "validation",
    "validation-labels": "held-out labelled",
    "random-walk": "random legal walk",
}
"""How ``parity.json`` names the population it measured, in English a reader can weigh."""
GREEDY_SUFFIX = "-greedy"
"""How the second sampling point of one checkpoint is named (``small`` / ``small-greedy``)."""
REPO_TYPE = "model"
ARCHITECTURES = {"decoder": "MoveDecoder", "encoder": "PositionEncoder"}
MODEL_TYPES = {"decoder": "rukh-move-decoder", "encoder": "rukh-position-encoder"}
ENCODER_HEADS = ("value", "blunder", "result")


class ModelPublishConfig(BaseConfig):
    """Where the staged folder goes and which links the card carries."""

    owner: str = "chorcat"
    publish_dir: str = "artifacts/publish"
    eval_dir: str = "artifacts/eval"
    demo_url: str = "https://rukh.borjaglez.com"
    course_url: str = "https://lab.rukh.borjaglez.com"
    repository_url: str = "https://github.com/borja-glez/rukh"
    license: str = "apache-2.0"
    datasets: list[str] = ["chorcat/rukh-games-1800", "chorcat/rukh-tokenizer"]


class RunSummary(BaseModel):
    """The part of an MLflow run a card needs."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    name: str | None = None
    params: dict[str, str] = {}
    metrics: dict[str, float] = {}


class ModelPublishResult(BaseModel):
    """What was staged and, unless this was a dry run, uploaded."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str
    repo_type: str = REPO_TYPE
    kind: str = "decoder"
    """``decoder`` or ``encoder``, read off the checkpoint's own weights."""
    stage: str
    dry_run: bool
    folder: str
    card_path: str
    files: list[str]
    weights_format: str
    """``safetensors`` or ``torch`` when ``safetensors`` is not installed."""
    run_id: str | None = None
    params: int
    tied_embeddings: bool = False
    """``lm_head.weight`` was left out of the weights file and is re-tied when loading."""


def _api():  # type: ignore[no-untyped-def]
    """The ``HfApi`` client (tests monkeypatch this)."""
    from huggingface_hub import HfApi

    return HfApi()


def _client():  # type: ignore[no-untyped-def]
    """The MLflow client on the local store (tests monkeypatch this)."""
    from mlflow.tracking import MlflowClient

    from rukh.tracking import tracking_uri

    return MlflowClient(tracking_uri=tracking_uri(create=False))


def read_run(run_id: str | None = None, run_name: str | None = None) -> RunSummary | None:
    """The MLflow run behind a checkpoint, by id or by the most recent run of that name.

    A missing store, a missing run or an MLflow that cannot be reached is not an error: the card
    simply falls back to what the checkpoint itself records.
    """
    from rukh.tracking import EXPERIMENT

    try:
        client = _client()
        if run_id:
            run = client.get_run(run_id)
        else:
            experiment = client.get_experiment_by_name(EXPERIMENT)
            if experiment is None or not run_name:
                return None
            found = client.search_runs(
                [experiment.experiment_id],
                filter_string=f"attributes.run_name = '{run_name}'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if not found:
                return None
            run = found[0]
    except Exception as exc:  # noqa: BLE001 - tracking is optional, never fatal for a release
        log.warning("could not read the MLflow run (%s: %s)", type(exc).__name__, exc)
        return None
    return RunSummary(
        run_id=run.info.run_id,
        name=run.info.run_name,
        params={str(k): str(v) for k, v in run.data.params.items()},
        metrics={str(k): float(v) for k, v in run.data.metrics.items()},
    )


DEMO_STAGES = {
    "tiny": "tiny-int8",
    "small": "small-fp16",
    "medium": "medium-fp16",
    "medium-dpo": "medium-dpo-fp16",
    "medium-grpo": "medium-grpo-fp16",
    "medium-elo": "medium-elo-fp16",
    "medium-lora": "medium-lora-fp16",
}
"""Stage ids the demo actually serves, keyed by the repository name without ``rukh-``.

The card used to link ``/?stage=<eval stage>``, and the demo does not know those names: an
unknown ``stage`` falls back to whatever the device would load anyway, so a card could promise a
game against a model the demo never loads. A repo the demo does not serve gets a bare link and a
sentence saying so (D-075).

The keys are the **repository** names and not the model size, because since M4 the demo serves
four different `medium` repositories and "medium" alone no longer says which one. Keeping this
list right is a promise to a reader who clicks: what they get has to be the weights the card is
about.
"""


def demo_link(cfg: ModelPublishConfig, repo_id: str) -> tuple[str, bool]:
    """``(url, served)``: where to send a reader, and whether the demo really runs this model."""
    name = repo_id.split("/")[-1].removeprefix("rukh-")
    stage = DEMO_STAGES.get(name)
    if stage is None:
        return cfg.demo_url, False
    return f"{cfg.demo_url}/?stage={stage}", True


def read_sweep(stage: str, cfg: ModelPublishConfig) -> dict[str, Any] | None:
    """The per-condition sweep of an Elo-conditioned stage, if one was run.

    A model whose whole point is that it answers ``<wXXXX>`` cannot be published with only the
    number it scores at one header. What the header *does* is the claim, and the claim is either
    measured on the card or it is marketing.
    """
    path = resolve(cfg.eval_dir) / f"{stage}-elo-sweep" / "results.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) and payload.get("rows") else None


def sweep_context(sweep: dict[str, Any] | None) -> dict[str, Any] | None:
    """The sweep as the card prints it: one row per condition, and the verdict in words."""
    if not sweep:
        return None
    rows = []
    for row in sweep["rows"]:
        interval = row.get("elo_ci")
        rows.append(
            {
                "header": f"<w{int(row['header_elo']):04d}>",
                "elo": "n/a" if row.get("elo") is None else f"{row['elo']:.0f}",
                "ci": (
                    f"{interval[0]:.0f}-{interval[1]:.0f}"
                    if isinstance(interval, list) and len(interval) == 2
                    else "n/a"
                ),
                "entropy": (
                    "n/a"
                    if row.get("first_move_entropy") is None
                    else f"{row['first_move_entropy']:.3f}"
                ),
                "legality": _percent(row.get("legality")),
                "top1": _percent(row.get("top1")),
            }
        )
    return {
        "rows": rows,
        "monotonic": bool(sweep.get("monotonic")),
        "separated": bool(sweep.get("separated")),
        "span": None if sweep.get("span") is None else f"{sweep['span']:.0f}",
    }


def read_eval(stage: str, cfg: ModelPublishConfig) -> dict[str, Any] | None:
    """The ``results.json`` the evaluation harness wrote for this stage, if it ran."""
    path = resolve(cfg.eval_dir) / stage / "results.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def check_eval_matches(ckpt: Path, stage: str, evaluation: dict[str, Any] | None) -> None:
    """Refuse to put one model's numbers on another model's card.

    ``read_eval`` finds results by stage name alone, so passing the wrong ``--stage`` renders a
    card whose weights and metrics come from different checkpoints, and nothing in the output
    says so. It nearly happened twice: once with the encoder (D-061) and once publishing
    ``small-v3`` under the stage that holds ``small`` v1's numbers. The guard added the first
    time only paired the two sampling points; this one compares what is actually being shipped
    against what was actually measured (D-074).

    The evaluation stores ``sha256`` of the checkpoint file, so ``best.pt`` and a ``step-*.pt``
    with identical weights are still different files and still refused. That is the safe
    direction: publish the file that was measured.
    """
    if not evaluation:
        return
    measured = str(evaluation.get("model_sha") or "")
    if not measured:
        return
    from rukh.eval.cache import file_sha

    actual = file_sha(Path(ckpt))
    if measured != actual:
        raise ValueError(
            f"stage {stage!r} was measured on a different checkpoint: its results.json records "
            f"model_sha {measured[:12]} and {Path(ckpt).as_posix()} hashes to {actual[:12]}. "
            f"Evaluate this checkpoint under this stage, or publish the one that was evaluated."
        )


def counterpart_stage(stage: str) -> str:
    """The stage that holds the same checkpoint at the other sampling setting (D-047).

    The Elo of a decoder is a property of the pair model+sampling, so the suite is run twice on
    one checkpoint and the two runs are stored side by side under ``<stage>`` and
    ``<stage>-greedy``. This is how the card finds the one it is not publishing.
    """
    return stage[: -len(GREEDY_SUFFIX)] if stage.endswith(GREEDY_SUFFIX) else stage + GREEDY_SUFFIX


def read_parity(folder: Path) -> dict[str, Any] | None:
    """The ``parity.json`` ``rukh export`` wrote next to the ONNX files, if it is there.

    Absent means "never measured" and the card says nothing about parity, which is the honest
    reading: a card that quietly omitted a bad number would be worse than one with no number.
    """
    path = Path(folder) / ONNX_DIR / PARITY_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        log.warning("%s is not readable JSON: the card goes out without the parity", path)
        return None
    return payload if isinstance(payload, dict) else None


def tied_names(model: Any) -> list[str]:
    """The tied heads of a model: the names whose storage is the embedding's, under another name.

    Asked of the tensors rather than of the config, because the encoder's tie lives one level
    down (``encoder.mlm_head.weight``) and only exists for the ``moves`` scheme.
    """
    state = model.state_dict()
    return [
        name
        for name, source in TIED_SOURCES.items()
        if name in state and source in state and state[name].data_ptr() == state[source].data_ptr()
    ]


def publish_state(model: Any) -> dict[str, Any]:
    """The state dict as it is published: on the CPU, contiguous and with no aliased tensor.

    With tied embeddings ``lm_head.weight`` *is* ``tokens.weight``; it is dropped here so the
    file holds every tensor exactly once. Loading re-ties it (see the module docstring).
    """
    dropped = set(tied_names(model))
    return {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if key not in dropped
    }


def write_weights(state: dict[str, Any], folder: Path) -> tuple[str, str]:
    """Write the state dict; returns ``(file name, format)``.

    ``safetensors`` is the format the Hub expects and the only one it will preview, so it wins
    when it is installed; otherwise the weights go out as a torch pickle under its own name
    rather than pretending to be something they are not. The state dict must already be free of
    aliases (``publish_state``): ``safetensors.save_file`` raises on two names for one storage.
    """
    import torch

    tensors = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    try:
        from safetensors.torch import save_file
    except ImportError:
        log.warning("safetensors is not installed; the weights go out as %s", TORCH_NAME)
        torch.save(tensors, folder / TORCH_NAME)
        return TORCH_NAME, "torch"
    save_file(tensors, str(folder / SAFETENSORS_NAME))
    return SAFETENSORS_NAME, "safetensors"


def write_config(
    payload: dict[str, Any],
    stage: str,
    params: int,
    folder: Path,
    kind: str = "decoder",
) -> dict[str, Any]:
    """Write ``config.json``: the model's shape plus the provenance of the checkpoint."""
    model_cfg = dict(payload.get("model_cfg") or {})
    scheme = str(model_cfg.get("input") or "moves")
    encoder = kind == "encoder"
    config = {
        "architectures": [ARCHITECTURES.get(kind, ARCHITECTURES["decoder"])],
        "model_type": MODEL_TYPES.get(kind, MODEL_TYPES["decoder"]),
        "library_name": "rukh",
        "rukh_version": __version__,
        "stage": stage,
        "step": payload.get("step"),
        "params": params,
        "tokenizer": scheme if encoder else "uci",
        "vocab_hash": payload.get("vocab_hash"),
        "data_manifest_sha": payload.get("data_manifest_sha"),
        "git_sha": payload.get("git_sha"),
        **(
            {
                "heads": list(ENCODER_HEADS),
                "pooling": _pooling(payload),
                "pretrained_from": _pretrained_from(payload),
            }
            if encoder
            else {}
        ),
        **model_cfg,
    }
    (folder / CONFIG_NAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return config


def _pooling(payload: dict[str, Any]) -> str:
    """How the published encoder pools its tokens, as its fine-tuning run recorded it."""
    return str((payload.get("cfg") or {}).get("pooling") or "mean")


def _pretrained_from(payload: dict[str, Any]) -> str | None:
    """The masked-move checkpoint the heads were fine-tuned from, or ``None`` for from scratch.

    ``encoder_ckpt`` is ``None`` in a perfectly legitimate run — training the encoder from
    scratch is the baseline the pretraining has to beat — so the card cannot claim masked move
    modeling unconditionally; this is the fact it gates that claim on.
    """
    found = (payload.get("cfg") or {}).get("encoder_ckpt")
    return str(found) if found else None


def write_vocab(folder: Path, kind: str, scheme: str) -> str:
    """Write the vocabulary the model actually reads, and return its path in the repository.

    A decoder (and an encoder on the ``moves`` scheme) reads the P1 UCI enumeration; an encoder
    on the ``squares`` scheme reads the 47 fixed square tokens, which are just as much part of
    the released model: without them the 69 integers of an input mean nothing.
    """
    if kind == "encoder" and scheme == "squares":
        from rukh.models.squares import SQUARE_TOKENS, SQUARE_VOCAB, vocab_hash

        path = folder / VOCAB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "scheme": "squares",
            "size": len(SQUARE_VOCAB),
            "sequence": SQUARE_TOKENS,
            "vocab_hash": vocab_hash(),
            "tokens": list(SQUARE_VOCAB),
        }
        path.write_text(
            json.dumps(payload, indent=0, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
        )
        return VOCAB_PATH
    UciTokenizer().export(folder / VOCAB_PATH)
    return VOCAB_PATH


def copy_onnx(onnx_dir: Path | None, folder: Path) -> list[str]:
    """Copy the exported ONNX files into ``onnx/`` and return what was copied.

    ``parity.json`` travels with them when the export measured parity. It is small, it is the
    evidence behind the agreement figures the card quotes, and a released file whose fidelity
    nobody can check is a claim rather than a measurement, so it is published too.
    """
    if onnx_dir is None:
        return []
    source = Path(onnx_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"no ONNX directory at {source}")
    target = folder / ONNX_DIR
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ONNX_FILES:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, target / name)
            copied.append(f"{ONNX_DIR}/{name}")
    if not copied:
        raise FileNotFoundError(f"{source} holds none of {', '.join(ONNX_FILES)}")
    if (source / PARITY_NAME).is_file():
        shutil.copy2(source / PARITY_NAME, target / PARITY_NAME)
        copied.append(f"{ONNX_DIR}/{PARITY_NAME}")
    return copied


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f} %"


def _sampled_label(sampled: dict[str, Any]) -> str:
    """The row name of the sampled legality rate, with the setting it was drawn under."""
    if not sampled or sampled.get("temperature") is None:
        return "Legality without the mask, sampled"
    top_k = sampled.get("top_k")
    tail = "" if top_k is None else f", top-k {top_k}"
    return f"Legality without the mask, sampled (T={float(sampled['temperature']):g}{tail})"


def _elo_cell(elo: dict[str, Any]) -> str:
    """The Elo row: an interval, or the one-sided bound of a separated fit."""
    if not elo:
        return "n/a"
    if elo.get("ci_low") is not None and elo.get("ci_high") is not None:
        return f"{elo['elo']:.0f} (95 % CI {elo['ci_low']:.0f}-{elo['ci_high']:.0f})"
    if elo.get("elo_lower") is not None:
        return f"> {elo['elo_lower']:.0f} (one-sided 95 % bound; every game won)"
    if elo.get("elo_upper") is not None:
        return f"< {elo['elo_upper']:.0f} (one-sided 95 % bound; every game lost)"
    return f"{elo['elo']:.0f} (no interval)"


def _drift(entry: dict[str, Any]) -> str:
    """The worst drift one precision showed, whichever output it was measured on."""
    for key in PARITY_DELTAS:
        if entry.get(key) is not None:
            return f"{float(entry[key]):.3g}"
    return "n/a"


def _run_name(name: object) -> str | None:
    """``lora-e4-20260920-160133`` -> ``lora-e4``: a card names an adapter, not a directory."""
    if not name:
        return None
    return re.sub(r"-\d{8}-\d{6}$", "", str(name))


def parity_context(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """The measured parity of the published ONNX files, as the card shows it.

    Every figure comes from ``parity.json``; nothing is recomputed here and nothing is assumed.
    ``None`` when there is no such file, so a card without the measurement stays silent about it
    rather than implying the files were checked.
    """
    precisions = (payload or {}).get("precisions")
    if not isinstance(precisions, dict) or payload is None:
        return None
    rows: list[dict[str, Any]] = []
    for name in PRECISIONS:
        entry = precisions.get(name)
        if not isinstance(entry, dict) or entry.get("agreement") is None:
            continue
        rate = float(entry["agreement"])
        rows.append(
            {
                "precision": name,
                "file": str(entry.get("file") or "n/a"),
                "rate": rate,
                "agreement": _percent(rate),
                "disagreement": _percent(1.0 - rate),
                "one_in": None if rate >= 1.0 else round(1.0 / (1.0 - rate)),
                "drift": _drift(entry),
                "met": rate >= PARITY_BAR,
            }
        )
    if not rows:
        return None
    adapted = payload.get("with_adapter") or {}
    adapted_rows = [
        {
            "precision": name,
            "agreement": _percent(float(entry["agreement"])),
            "drift": _drift(entry),
        }
        for name in PRECISIONS
        if isinstance(entry := (adapted.get("precisions") or {}).get(name), dict)
        and entry.get("agreement") is not None
    ]
    return {
        "rows": rows,
        "below": [row for row in rows if not row["met"]],
        "perfect": all(row["rate"] >= 1.0 for row in rows),
        "positions": payload.get("positions"),
        "source": PARITY_SOURCES.get(str(payload.get("source")), str(payload.get("source"))),
        "measures": payload.get("measures"),
        "exporter": payload.get("exporter"),
        "bar": _percent(PARITY_BAR),
        "warning": payload.get("warning"),
        # A graph that takes its LoRA factors as inputs makes two claims, not one: fed zeros it is
        # the checkpoint, and fed a real adapter it is that style. Both are in `parity.json`.
        "adapter_inputs": bool(payload.get("adapter_inputs")),
        "adapter_name": _run_name(adapted.get("adapter")),
        "adapter_rows": adapted_rows,
    }


def acceptance_bars(evaluation: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The decoder's two bars from ``GOAL.md``, each against the value that was measured.

    A bar whose metric is missing is left out of the list entirely: "not met" and "not measured"
    are different statements, and only one of them would be true.
    """
    from rukh.eval.suite import GOAL_ELO, GOAL_LEGALITY

    measured = evaluation or {}
    argmax = measured.get("legality_argmax") or {}
    elo = measured.get("elo") or {}
    bars: list[dict[str, Any]] = []
    rate = argmax.get("rate")
    if rate is not None:
        bars.append(
            {
                "id": "legality",
                "name": "Legality without the mask, argmax",
                "target": f"at least {GOAL_LEGALITY * 100:.0f} %",
                "measured": _percent(rate),
                "met": float(rate) >= GOAL_LEGALITY,
            }
        )
    rating = elo.get("elo")
    if rating is not None:
        bars.append(
            {
                "id": "elo",
                "name": "Estimated Elo",
                "target": f"at least {GOAL_ELO:.0f}",
                "measured": _elo_cell(elo),
                "met": float(rating) >= GOAL_ELO,
            }
        )
    return bars


def encoder_acceptance_bars(evaluation: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The encoder's two bars from ``GOAL.md``, read on the same metrics the harness reads.

    The value bar is read on **Spearman** and not on Pearson, for the reason
    ``rukh.eval.encoder.GOAL_VALUE_CORRELATION`` spells out; the card must not be able to pass a
    criterion the evaluation fails by picking the friendlier of the two correlations.
    """
    from rukh.eval.encoder import GOAL_MARGIN, GOAL_VALUE_CORRELATION

    measured = evaluation or {}
    bars: list[dict[str, Any]] = []
    margin = measured.get("f1_margin")
    if margin is not None:
        bars.append(
            {
                "id": "blunder",
                "name": "Blunder F1 over the material baseline",
                "target": f"at least +{GOAL_MARGIN:.0f} F1 points",
                "measured": f"{float(margin):+.1f} F1 points",
                "met": float(margin) >= GOAL_MARGIN,
            }
        )
    spearman = (measured.get("encoder_value") or {}).get("spearman")
    if spearman is not None:
        bars.append(
            {
                "id": "value",
                "name": "Value vs Stockfish cp, Spearman",
                "target": f"at least {GOAL_VALUE_CORRELATION:.2f}",
                "measured": _ratio(spearman),
                "met": float(spearman) >= GOAL_VALUE_CORRELATION,
            }
        )
    return bars


def _operating_point(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    """One (Elo, sampling) pair: the rating and the setting the games were played at."""
    measured = evaluation or {}
    elo = measured.get("elo") or {}
    if elo.get("elo") is None:
        return None
    config = measured.get("config") or {}
    sampled = measured.get("legality_sampled") or {}
    temperature = config.get("temperature", sampled.get("temperature"))
    if temperature is None:
        return None
    top_k = config.get("top_k", sampled.get("top_k"))
    tail = "" if top_k is None else f", top-k {int(top_k)}"
    return {
        "stage": measured.get("stage"),
        "elo": f"{float(elo['elo']):.0f}",
        "cell": _elo_cell(elo),
        "temperature": float(temperature),
        "setting": f"temperature {float(temperature):g}{tail}",
    }


def sampling_context(
    evaluation: dict[str, Any] | None, counterpart: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The same weights at a second sampling setting, when one was measured (D-047).

    Two evaluations describe one model only when they ran on the same tensors, so the checkpoint
    hashes have to match: without that check the card would compare two models and call the
    difference a property of the sampling.
    """
    here = _operating_point(evaluation)
    there = _operating_point(counterpart)
    sha = (evaluation or {}).get("model_sha")
    if here is None or there is None or not sha:
        return None
    if (counterpart or {}).get("model_sha") != sha:
        return None
    if here["temperature"] == there["temperature"]:
        return None
    strong, varied = sorted((here, there), key=lambda point: float(point["temperature"]))
    return {"model_sha": str(sha)[:12], "strong": strong, "varied": varied, "published": here}


def card_context(
    repo_id: str,
    stage: str,
    cfg: ModelPublishConfig,
    config: dict[str, Any],
    evaluation: dict[str, Any] | None,
    run: RunSummary | None,
    files: list[str],
    parity: dict[str, Any] | None = None,
    counterpart: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the Jinja card needs, with every absent metric spelled ``n/a``."""
    elo = (evaluation or {}).get("elo") or {}
    accuracy = (evaluation or {}).get("accuracy") or {}
    argmax = (evaluation or {}).get("legality_argmax") or {}
    sampled = (evaluation or {}).get("legality_sampled") or {}
    puzzles = (evaluation or {}).get("puzzles") or {}
    metrics = [
        ("Legality without the mask, argmax", _percent(argmax.get("rate"))),
        (_sampled_label(sampled), _percent(sampled.get("rate"))),
        ("Top-1 next move", _percent(accuracy.get("top1"))),
        ("Top-3 next move", _percent(accuracy.get("top3"))),
        ("Puzzles solved", _percent(puzzles.get("rate"))),
        ("Estimated Elo", _elo_cell(elo)),
    ]
    bands = [
        (band["band"], _percent(band["rate"]))
        for band in puzzles.get("bands", [])
        if isinstance(band, dict)
    ]
    recipe = run.params if run else {}
    notes = [str(note) for note in (evaluation or {}).get("notes", []) if str(note).strip()]
    return {
        "notes": notes,
        "tied_embeddings": bool(config.get("tie_embeddings")),
        "repo_id": repo_id,
        "stage": stage,
        "license": cfg.license,
        "datasets": cfg.datasets,
        "demo_url": demo_link(cfg, repo_id)[0],
        "demo_serves_this": demo_link(cfg, repo_id)[1],
        "course_url": cfg.course_url,
        "repository_url": cfg.repository_url,
        "params": config.get("params", 0),
        "config": config,
        "config_json": json.dumps(config, indent=2, ensure_ascii=False),
        "metrics": metrics,
        "sweep": sweep_context(read_sweep(stage, cfg)),
        "bars": acceptance_bars(evaluation),
        "parity": parity_context(parity),
        "sampling": sampling_context(evaluation, counterpart),
        "puzzle_bands": bands,
        "evaluated_on": (evaluation or {}).get("date"),
        "suite": (evaluation or {}).get("suite"),
        "run_id": run.run_id if run else None,
        "recipe": sorted(recipe.items()),
        "files": files,
        "has_onnx": any(name.startswith(f"{ONNX_DIR}/") for name in files),
        "rukh_version": __version__,
    }


def _ratio(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def encoder_card_context(
    repo_id: str,
    stage: str,
    cfg: ModelPublishConfig,
    config: dict[str, Any],
    evaluation: dict[str, Any] | None,
    run: RunSummary | None,
    files: list[str],
    parity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the encoder card needs: its metrics, the baseline's, and the input scheme."""
    from rukh.eval.encoder import base_rate_caveat, sentence

    measured = evaluation or {}
    encoder = measured.get("encoder_blunder") or {}
    fixed = measured.get("encoder_blunder_fixed") or {}
    ranked = measured.get("encoder_blunder_ranking") or {}
    baseline = measured.get("heuristic_blunder") or {}
    value = measured.get("encoder_value") or {}
    margin = measured.get("f1_margin")
    tuned = measured.get("threshold_tuned")
    configured = measured.get("threshold_fixed")
    metrics = [
        (
            "Blunder F1" + ("" if tuned is None else f" (tuned, `p >= {float(tuned):.4g}`)"),
            _percent(encoder.get("f1")),
        ),
        ("Blunder precision", _percent(encoder.get("precision"))),
        ("Blunder recall", _percent(encoder.get("recall"))),
        (
            "Blunder F1"
            + ("" if configured is None else f" (fixed, `p >= {float(configured):.4g}`)"),
            _percent(fixed.get("f1")),
        ),
        ("Blunder ROC AUC", _ratio(ranked.get("roc_auc"))),
        ("Blunder average precision", _ratio(ranked.get("average_precision"))),
        ("Blunder base rate", _percent(measured.get("blunder_base_rate"))),
        ("Blunder F1, material baseline", _percent(baseline.get("f1"))),
        (
            "Margin over the baseline",
            "n/a" if margin is None else f"{float(margin):+.1f} F1 points",
        ),
        ("Value vs Stockfish cp, Pearson", _ratio(value.get("pearson"))),
        ("Value vs Stockfish cp, Spearman", _ratio(value.get("spearman"))),
        ("Result accuracy", _percent(measured.get("result_accuracy"))),
    ]
    curve = [
        (f"{float(point.get('fraction', 0)) * 100:.0f} %", point.get("train_labels"))
        for point in measured.get("label_curve", [])
        if isinstance(point, dict)
    ]
    return {
        "repo_id": repo_id,
        "stage": stage,
        "license": cfg.license,
        "datasets": cfg.datasets,
        "demo_url": demo_link(cfg, repo_id)[0],
        "demo_serves_this": demo_link(cfg, repo_id)[1],
        "course_url": cfg.course_url,
        "repository_url": cfg.repository_url,
        "params": config.get("params", 0),
        "config": config,
        "config_json": json.dumps(config, indent=2, ensure_ascii=False),
        "scheme": config.get("tokenizer", "squares"),
        "pooling": config.get("pooling", "mean"),
        "pretrained_from": config.get("pretrained_from"),
        "heads": list(config.get("heads", ENCODER_HEADS)),
        "metrics": metrics,
        "bars": encoder_acceptance_bars(evaluation),
        "parity": parity_context(parity),
        "curve": curve,
        "positions": measured.get("items"),
        "blunder_items": measured.get("blunder_items"),
        "blunder_caveat": sentence(base_rate_caveat(measured.get("blunder_base_rate"))),
        "tune_items": measured.get("tune_items"),
        "score_items": measured.get("score_items"),
        "tune_games": measured.get("tune_games"),
        "score_games": measured.get("score_games"),
        "threshold_tuned": tuned,
        "evaluated_on": measured.get("date"),
        "notes": [str(note) for note in measured.get("notes", []) if str(note).strip()],
        "run_id": run.run_id if run else None,
        "recipe": sorted((run.params if run else {}).items()),
        "files": files,
        "has_onnx": any(name.startswith(f"{ONNX_DIR}/") for name in files),
        "rukh_version": __version__,
    }


def pretrained_encoder_card_context(
    repo_id: str,
    stage: str,
    cfg: ModelPublishConfig,
    config: dict[str, Any],
    payload: dict[str, Any],
    run: RunSummary | None,
    files: list[str],
) -> dict[str, Any]:
    """The card of a masked-move pretraining run: what it measured, and what starts from it.

    A pretraining checkpoint has no blunder F1 and no value correlation to report; rendering it
    through the heads' card would say "trained from scratch on Stockfish labels" about a model
    that never saw a label. What it has is the held-out masked-move loss the run selected
    ``best.pt`` on, which the checkpoint carries, and the top-1 of the same evaluation when the
    MLflow run is reachable.
    """
    metrics = run.metrics if run else {}
    top1 = next((metrics[key] for key in ("val_top1", "val/top1") if key in metrics), None)
    best_val = payload.get("best_val")
    return {
        "repo_id": repo_id,
        "stage": stage,
        "license": cfg.license,
        "datasets": cfg.datasets,
        "course_url": cfg.course_url,
        "repository_url": cfg.repository_url,
        "params": config.get("params", 0),
        "config": config,
        "config_json": json.dumps(config, indent=2, ensure_ascii=False),
        # The model whose heads were fine-tuned from this one: the course publishes it under the
        # plain name, and this checkpoint is only on the Hub so that one can be rebuilt.
        "heads_repo": f"{cfg.owner}/rukh-encoder",
        "best_val": "n/a" if best_val is None else f"{float(best_val):.4f}",
        "val_top1": _percent(top1),
        "step": config.get("step") or "n/a",
        "run_id": run.run_id if run else None,
        "recipe": sorted((run.params if run else {}).items()),
        "files": files,
        "rukh_version": __version__,
    }


def render_card(context: dict[str, Any], template: str = CARD_TEMPLATE) -> str:
    """Render the English model card."""
    env = Environment(
        loader=FileSystemLoader(str(CARDS_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env.get_template(template).render(**context)


def publish_model(
    ckpt: Path,
    repo: str,
    cfg: ModelPublishConfig | None = None,
    onnx_dir: Path | None = None,
    stage: str | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
) -> ModelPublishResult:
    """Stage (and unless ``dry_run``, upload) one trained model as a Hub model repository."""
    from rukh.train import load_any

    cfg = cfg or ModelPublishConfig()
    ckpt = Path(ckpt)
    repo_id = repo if "/" in repo else f"{cfg.owner}/{repo}"
    name = stage or repo_id.split("/")[-1].removeprefix("rukh-")
    model, payload, kind = load_any(ckpt)
    state = publish_state(model)
    params = (
        model.num_params(non_embedding=False)
        if hasattr(model, "num_params")
        else sum(parameter.numel() for parameter in model.parameters())
    )

    folder = resolve(cfg.publish_dir) / repo_id
    folder.mkdir(parents=True, exist_ok=True)
    weights_name, weights_format = write_weights(state, folder)
    config = write_config(payload, name, params, folder, kind)
    write_vocab(folder, kind, str(config.get("input") or "moves"))
    files = [weights_name, CONFIG_NAME, VOCAB_PATH, *copy_onnx(onnx_dir, folder)]

    run = read_run(run_id, run_name=ckpt.parent.name)
    evaluation = read_eval(name, cfg)
    check_eval_matches(ckpt, name, evaluation)
    measured_parity = read_parity(folder)
    from rukh.models import PositionEncoder

    if isinstance(model, PositionEncoder):
        # Pretraining only: no heads on it, so none of the heads' numbers apply.
        card = render_card(
            pretrained_encoder_card_context(repo_id, name, cfg, config, payload, run, files),
            PRETRAINED_ENCODER_CARD_TEMPLATE,
        )
    elif kind == "encoder":
        card = render_card(
            encoder_card_context(
                repo_id, name, cfg, config, evaluation, run, files, measured_parity
            ),
            ENCODER_CARD_TEMPLATE,
        )
    else:
        card = render_card(
            card_context(
                repo_id,
                name,
                cfg,
                config,
                evaluation,
                run,
                files,
                measured_parity,
                read_eval(counterpart_stage(name), cfg),
            )
        )
    card_path = folder / README_NAME
    card_path.write_text(card, encoding="utf-8", newline="\n")

    if not dry_run:
        api = _api()
        api.create_repo(repo_id, repo_type=REPO_TYPE, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(folder),
            repo_type=REPO_TYPE,
            commit_message=f"Publish {name}",
        )
    return ModelPublishResult(
        repo_id=repo_id,
        kind=kind,
        stage=name,
        dry_run=dry_run,
        folder=folder.as_posix(),
        card_path=card_path.as_posix(),
        files=[*files, README_NAME],
        weights_format=weights_format,
        run_id=run.run_id if run else None,
        params=params,
        tied_embeddings=bool(tied_names(model)),
    )
