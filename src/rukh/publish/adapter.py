"""Publishing a LoRA adapter, which is not a model and must not pretend to be one.

A model repository in this project carries weights, three ONNX exports, a vocabulary and a card
whose numbers come from ``rukh eval``. An adapter has none of that. It is two matrices per adapted
projection -- 1.6 MB against the 440 MB of the weights it mounts on -- and on its own it does
nothing at all: without the exact base checkpoint it is a pile of numbers with no model to correct.

So the card leads with the base model and the command that loads it, and the metrics it publishes
are the ones that say what the adapter *changed*: the share of first moves it shifted, which is the
whole visible point of a style adapter, and the Elo of base-plus-adapter against the base alone,
which is what that style cost in strength. Publishing an adapter with the base model's Elo and no
mention of the adapter's own effect would be publishing somebody else's number.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rukh.config import BaseConfig
from rukh.models.lora import ADAPTER_CONFIG, ADAPTER_FILE, LoraConfig
from rukh.paths import resolve
from rukh.publish.model import README_NAME, ModelPublishConfig, _api, read_eval, render_card

log = logging.getLogger(__name__)

__all__ = ["AdapterPublishResult", "publish_adapter", "publish_qwen_adapter"]

REPO_TYPE = "model"
ADAPTER_CARD_TEMPLATE = "adapter.md.jinja"


class AdapterEffect(BaseConfig):
    """What the adapter measurably did, as opposed to what it was meant to do."""

    first_move: str | None = None
    """UCI of the opening move the style is about, e.g. ``e2e4``."""
    share_before: float | None = None
    share_after: float | None = None
    """Share of self-play games opening with ``first_move``, base and adapted."""
    elo_base: float | None = None
    elo_adapted: float | None = None
    games: int | None = None
    """Self-play games behind the two shares."""
    train_games: int | None = None
    """Games in the style slice the adapter was trained on."""
    predicate: str | None = None
    """How that slice was selected, verbatim from the style manifest."""


class AdapterPublishResult(BaseModel):
    """What was staged and, unless this was a dry run, uploaded."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str
    base_repo: str
    stage: str
    dry_run: bool
    folder: str
    card_path: str
    files: list[str]
    params: int
    bytes: int


def adapter_params(lora: LoraConfig, n_layer: int, d_model: int) -> int:
    """``2 * r * d_model`` per adapted matrix per layer: the number the card leads with."""
    return n_layer * len(lora.targets) * 2 * lora.r * d_model


def publish_adapter(
    run_dir: Path | str,
    repo: str,
    base_repo: str,
    cfg: ModelPublishConfig | None = None,
    stage: str | None = None,
    effect: AdapterEffect | None = None,
    base_config: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> AdapterPublishResult:
    """Stage (and unless ``dry_run``, upload) one LoRA adapter as its own small repository."""
    cfg = cfg or ModelPublishConfig()
    source = Path(run_dir)
    weights = source / ADAPTER_FILE
    if not weights.is_file():
        raise FileNotFoundError(f"no adapter at {weights}")
    lora = LoraConfig.model_validate_json((source / ADAPTER_CONFIG).read_text(encoding="utf-8"))

    repo_id = repo if "/" in repo else f"{cfg.owner}/{repo}"
    base_id = base_repo if "/" in base_repo else f"{cfg.owner}/{base_repo}"
    name = stage or repo_id.split("/")[-1].removeprefix("rukh-")

    folder = resolve(cfg.publish_dir) / repo_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ADAPTER_FILE).write_bytes(weights.read_bytes())
    (folder / ADAPTER_CONFIG).write_text(
        json.dumps(
            {"lora": lora.model_dump(mode="json"), "base_model": base_id, "format": "rukh-lora-1"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    config = base_config or {}
    params = adapter_params(lora, int(config.get("n_layer", 16)), int(config.get("d_model", 768)))
    card = render_card(
        {
            "repo_id": repo_id,
            "base_repo": base_id,
            "stage": name,
            "license": cfg.license,
            "lora": lora.model_dump(mode="json"),
            "targets": ", ".join(lora.targets),
            "params": params,
            "bytes": weights.stat().st_size,
            "effect": (effect or AdapterEffect()).model_dump(mode="json"),
            "demo_url": cfg.demo_url,
            "course_url": cfg.course_url,
            "repository_url": cfg.repository_url,
            "datasets": cfg.datasets,
        },
        ADAPTER_CARD_TEMPLATE,
    )
    card_path = folder / README_NAME
    card_path.write_text(card, encoding="utf-8", newline="\n")

    files = [ADAPTER_FILE, ADAPTER_CONFIG, README_NAME]
    if not dry_run:
        api = _api()
        api.create_repo(repo_id, repo_type=REPO_TYPE, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(folder),
            repo_type=REPO_TYPE,
            commit_message=f"Publish {name}",
        )
    return AdapterPublishResult(
        repo_id=repo_id,
        base_repo=base_id,
        stage=name,
        dry_run=dry_run,
        folder=folder.as_posix(),
        card_path=card_path.as_posix(),
        files=files,
        params=params,
        bytes=weights.stat().st_size,
    )


QWEN_CARD_TEMPLATE = "qwen-adapter.md.jinja"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f} %"


def qwen_card_context(
    repo_id: str,
    run_dir: Path,
    cfg: ModelPublishConfig,
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Everything the Qwen adapter's card says, read off the run and the evaluation.

    Nothing here is written by hand. ``run.json`` is what ``rukh train qwen`` recorded about
    itself -- including whether four bits were actually used -- and the evaluation is the report
    of ``rukh eval qwen``, which ran the project's own harness. A card that quoted the intended
    recipe instead of the executed one would be the wrong kind of document.
    """
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    peft_config = json.loads((run_dir / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    results: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    if evaluation:
        written = evaluation.get("written") or {}
        asked = int(written.get("asked") or 0)

        def share(count: int) -> str:
            return f"{100 * count / asked:.2f} %" if asked else "n/a"

        elo = evaluation.get("elo") or {}
        results = [
            ("Legal moves written, no mask", share(int(written.get("legal") or 0))),
            ("Top-1 next move", _percent(evaluation.get("top1"))),
            ("Puzzles solved", _percent((evaluation.get("puzzles") or {}).get("rate"))),
            ("Estimated Elo", _elo_text(elo)),
        ]
        failures = [
            ("legal", share(int(written.get("legal") or 0))),
            ("illegal here", share(int(written.get("illegal") or 0))),
            ("not a move at all", share(int(written.get("unparseable") or 0))),
            ("ambiguous SAN", share(int(written.get("ambiguous") or 0))),
            ("nothing written", share(int(written.get("empty") or 0))),
        ]
    return {
        "repo_id": repo_id,
        "base_model": str(peft_config.get("base_model_name_or_path", run.get("model"))),
        "license": cfg.license,
        "lora": peft_config,
        "targets": ", ".join(sorted(peft_config.get("target_modules") or [])),
        "trainable_params": int(run.get("trainable_params") or 0),
        "total_params": int(run.get("total_params") or 1),
        "train_samples": int(run.get("train_samples") or 0),
        "steps": int(run.get("steps") or 0),
        "four_bit": bool(run.get("four_bit")),
        "weights_memory_mb": run.get("weights_memory_mb"),
        "peak_memory_mb": run.get("peak_memory_mb"),
        "results": results,
        "failures": failures,
        "course_url": cfg.course_url,
        "repository_url": cfg.repository_url,
    }


def _elo_text(elo: dict[str, Any]) -> str:
    if not elo:
        return "n/a"
    if elo.get("separated"):
        if elo.get("elo_lower") is not None:
            return f"> {elo['elo_lower']:.0f} (one-sided 95 % bound; every game won)"
        if elo.get("elo_upper") is not None:
            return f"< {elo['elo_upper']:.0f} (one-sided 95 % bound; every game lost)"
    low, high = elo.get("ci_low"), elo.get("ci_high")
    if low is None or high is None:
        return f"{elo.get('elo', 0):.0f} (no interval)"
    return f"{elo['elo']:.0f} (95 % CI {low:.0f}-{high:.0f})"


def publish_qwen_adapter(
    run_dir: Path | str,
    repo: str,
    cfg: ModelPublishConfig | None = None,
    stage: str = "qwen3-pgn-qlora",
    dry_run: bool = False,
) -> AdapterPublishResult:
    """Upload a ``peft`` adapter folder as it stands, plus a card built from the run's own record.

    Unlike the project's own adapters this one is **not** re-staged: ``peft`` already wrote a
    valid ``adapter_config.json`` and ``adapter_model.safetensors``, and the Hub knows that
    format. Rewriting them into our format would make the file unusable with the two lines of
    ``peft`` that every reader would actually type.
    """
    cfg = cfg or ModelPublishConfig()
    source = Path(run_dir)
    weights = source / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"no peft adapter at {weights}")
    repo_id = repo if "/" in repo else f"{cfg.owner}/{repo}"

    evaluation = read_eval(stage, cfg)
    context = qwen_card_context(repo_id, source, cfg, evaluation)
    (source / README_NAME).write_text(
        render_card(context, QWEN_CARD_TEMPLATE), encoding="utf-8", newline="\n"
    )
    files = sorted(path.name for path in source.iterdir() if path.is_file())

    if not dry_run:
        api = _api()
        api.create_repo(repo_id, repo_type=REPO_TYPE, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(source),
            repo_type=REPO_TYPE,
            commit_message=f"Publish {stage}",
        )
    return AdapterPublishResult(
        repo_id=repo_id,
        base_repo=str(context["base_model"]),
        stage=stage,
        dry_run=dry_run,
        folder=source.as_posix(),
        card_path=(source / README_NAME).as_posix(),
        files=files,
        params=int(context["trainable_params"]),
        bytes=weights.stat().st_size,
    )
