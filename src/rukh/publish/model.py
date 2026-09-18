"""Publish a trained decoder to the Hugging Face Hub: weights, ONNX, tokenizer and card.

One repository per stage holds everything somebody needs to reproduce or run the model: the
PyTorch weights, the fp16 and int8 ONNX files the demo loads, the exact vocabulary the model was
trained on, and an English card written from the MLflow run and the evaluation ``results.json``
rather than by hand, so the numbers in the card are the numbers that were measured.

The folder is always staged locally first (under ``artifacts/publish/<repo>/``) and uploaded in
one ``upload_folder`` call; ``dry_run`` stops after staging and touches no network. Every
``HfApi`` call goes through ``_api`` so the tests can replace it, exactly as ``data/publish``
does.

The head is tied to the token embedding, so ``lm_head.weight`` and ``tokens.weight`` are one
tensor under two names. ``safetensors`` refuses to serialise that (it stores tensors, not
aliases) and counting both would inflate the parameter count by a whole embedding table, so the
tied name is dropped from what is written and the loader re-ties it: ``MoveDecoder`` builds the
tie in its constructor and ``rukh.train.load_state`` accepts the missing name.
"""

from __future__ import annotations

import json
import logging
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
from rukh.train.checkpoint import TIED_HEAD

log = logging.getLogger(__name__)

CARD_TEMPLATE = "model.md.jinja"
CONFIG_NAME = "config.json"
README_NAME = "README.md"
SAFETENSORS_NAME = "model.safetensors"
TORCH_NAME = "pytorch_model.bin"
VOCAB_PATH = "tokenizer/vocab.json"
ONNX_DIR = "onnx"
ONNX_FILES = ("model-fp16.onnx", "model-int8.onnx", "model.onnx")
REPO_TYPE = "model"


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


def publish_state(model: Any) -> dict[str, Any]:
    """The state dict as it is published: on the CPU, contiguous and with no aliased tensor.

    With tied embeddings ``lm_head.weight`` *is* ``tokens.weight``; it is dropped here so the
    file holds every tensor exactly once. Loading re-ties it (see the module docstring).
    """
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    if model.cfg.tie_embeddings:
        state.pop(TIED_HEAD, None)
    return state


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


def write_config(payload: dict[str, Any], stage: str, params: int, folder: Path) -> dict[str, Any]:
    """Write ``config.json``: the decoder shape plus the provenance of the checkpoint."""
    model_cfg = dict(payload.get("model_cfg") or {})
    config = {
        "architectures": ["MoveDecoder"],
        "model_type": "rukh-move-decoder",
        "library_name": "rukh",
        "rukh_version": __version__,
        "stage": stage,
        "step": payload.get("step"),
        "params": params,
        "tokenizer": "uci",
        "vocab_hash": payload.get("vocab_hash"),
        "data_manifest_sha": payload.get("data_manifest_sha"),
        "git_sha": payload.get("git_sha"),
        **model_cfg,
    }
    (folder / CONFIG_NAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return config


def copy_onnx(onnx_dir: Path | None, folder: Path) -> list[str]:
    """Copy the exported ONNX files into ``onnx/`` and return what was copied."""
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


def card_context(
    repo_id: str,
    stage: str,
    cfg: ModelPublishConfig,
    config: dict[str, Any],
    evaluation: dict[str, Any] | None,
    run: RunSummary | None,
    files: list[str],
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
        "demo_url": f"{cfg.demo_url}/?stage={stage}",
        "course_url": cfg.course_url,
        "repository_url": cfg.repository_url,
        "params": config.get("params", 0),
        "config": config,
        "config_json": json.dumps(config, indent=2, ensure_ascii=False),
        "metrics": metrics,
        "puzzle_bands": bands,
        "evaluated_on": (evaluation or {}).get("date"),
        "suite": (evaluation or {}).get("suite"),
        "run_id": run.run_id if run else None,
        "recipe": sorted(recipe.items()),
        "files": files,
        "has_onnx": any(name.startswith(f"{ONNX_DIR}/") for name in files),
        "rukh_version": __version__,
    }


def render_card(context: dict[str, Any]) -> str:
    """Render the English model card."""
    env = Environment(
        loader=FileSystemLoader(str(CARDS_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env.get_template(CARD_TEMPLATE).render(**context)


def publish_model(
    ckpt: Path,
    repo: str,
    cfg: ModelPublishConfig | None = None,
    onnx_dir: Path | None = None,
    stage: str | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
) -> ModelPublishResult:
    """Stage (and unless ``dry_run``, upload) one trained decoder as a Hub model repository."""
    from rukh.train import load_model

    cfg = cfg or ModelPublishConfig()
    ckpt = Path(ckpt)
    repo_id = repo if "/" in repo else f"{cfg.owner}/{repo}"
    name = stage or repo_id.split("/")[-1].removeprefix("rukh-")
    model, payload = load_model(ckpt)
    state = publish_state(model)
    params = model.num_params(non_embedding=False)

    folder = resolve(cfg.publish_dir) / repo_id
    folder.mkdir(parents=True, exist_ok=True)
    weights_name, weights_format = write_weights(state, folder)
    config = write_config(payload, name, params, folder)
    UciTokenizer().export(folder / VOCAB_PATH)
    files = [weights_name, CONFIG_NAME, VOCAB_PATH, *copy_onnx(onnx_dir, folder)]

    run = read_run(run_id, run_name=ckpt.parent.name)
    evaluation = read_eval(name, cfg)
    card = render_card(card_context(repo_id, name, cfg, config, evaluation, run, files))
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
        stage=name,
        dry_run=dry_run,
        folder=folder.as_posix(),
        card_path=card_path.as_posix(),
        files=[*files, README_NAME],
        weights_format=weights_format,
        run_id=run.run_id if run else None,
        params=params,
        tied_embeddings=model.cfg.tie_embeddings,
    )
