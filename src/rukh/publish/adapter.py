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
from rukh.publish.model import README_NAME, ModelPublishConfig, _api, render_card

log = logging.getLogger(__name__)

__all__ = ["AdapterPublishResult", "publish_adapter"]

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
