"""``rukh publish cards``: every card regenerated from what is on disk, and only the card uploaded.

A card is a rendering of the run, the evaluation and the links of the day it was written. When
the table changes -- a nightly, a corrected ladder, a new lesson URL -- the card is stale, and
republishing 460 MB of weights to change a paragraph is the wrong tool. This stages every
repository of the catalogue the way its own publisher does (``dry_run``: nothing leaves the
machine) and then uploads ``README.md`` alone with ``upload_file``.

It needs the checkpoints on disk, which is what ``rukh pull`` is for, and the evaluations in
``artifacts/eval``, which is what ``rukh eval nightly`` leaves behind.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rukh import paths
from rukh.hub import Artefact, catalogue
from rukh.publish.model import README_NAME, ModelPublishConfig, _api
from rukh.train.checkpoint import resolve_run

log = logging.getLogger(__name__)

ONNX_DIRS: dict[str, str] = {
    "tiny": "artifacts/onnx/tiny",
    "small": "artifacts/onnx/small-v3",
    "medium-v4": "artifacts/onnx/medium-v4",
    "encoder-v4": "artifacts/onnx/encoder-v4",
    "medium-elo": "artifacts/onnx/medium-elo",
    "medium-masters": "artifacts/onnx/medium-masters",
    "medium-v4-dpo-onpolicy": "artifacts/onnx/medium-dpo",
    "medium-v4-grpo": "artifacts/onnx/medium-grpo",
}
"""The ONNX export each model's card reports parity from, when it is on disk."""

EFFECTS_DIR = "artifacts/publish/effects"


class CardRefresh(BaseModel):
    """One repository's card: where it was staged and whether it went up."""

    model_config = ConfigDict(extra="forbid")

    name: str
    repo_id: str
    repo_type: str
    card: str | None = None
    status: str
    """``staged`` (dry run), ``uploaded`` or ``failed``."""
    detail: str | None = None


def stage_card(artefact: Artefact, cfg: ModelPublishConfig) -> Path:
    """Render ``artefact``'s card through its own publisher, in dry-run mode; returns the card."""
    target = resolve_run(paths.resolve(artefact.target))
    if artefact.kind == "dataset":
        from rukh.data.publish import PublishConfig, publish

        result = publish(artefact.name, PublishConfig(owner=cfg.owner), dry_run=True)
        return Path(result.card_path)
    if artefact.kind == "adapter":
        from rukh.publish.adapter import AdapterEffect, publish_adapter

        effect_path = paths.resolve(EFFECTS_DIR) / f"{artefact.name}.json"
        effect = (
            AdapterEffect.model_validate_json(effect_path.read_text(encoding="utf-8"))
            if effect_path.is_file()
            else None
        )
        result = publish_adapter(
            target, artefact.repo_id, f"{cfg.owner}/rukh-medium", cfg, effect=effect, dry_run=True
        )
        return Path(result.card_path)
    if artefact.kind == "peft":
        from rukh.publish.adapter import publish_qwen_adapter

        result = publish_qwen_adapter(target, artefact.repo_id, cfg, dry_run=True)
        return Path(result.card_path)
    if artefact.kind == "reward":
        from rukh.publish.reward import publish_reward

        result = publish_reward(target, artefact.repo_id, cfg, dry_run=True)
        return Path(result.card_path)
    from rukh.publish.model import publish_model

    onnx = ONNX_DIRS.get(artefact.name)
    onnx_dir = paths.resolve(onnx) if onnx and paths.resolve(onnx).is_dir() else None
    result = publish_model(
        target, artefact.repo_id, cfg, onnx_dir=onnx_dir, stage=artefact.stage, dry_run=True
    )
    return Path(result.card_path)


def refresh_cards(
    only: set[str] | None = None,
    dry_run: bool = False,
    cfg: ModelPublishConfig | None = None,
    api: Any | None = None,
) -> list[CardRefresh]:
    """Regenerate the card of every repository (or of ``only``); upload it unless ``dry_run``."""
    cfg = cfg or ModelPublishConfig()
    wanted = [a for a in catalogue() if not only or a.name in only or a.repo_id in only]
    client = None
    outcomes: list[CardRefresh] = []
    for artefact in wanted:
        repo_type = artefact.hub_repo_type
        outcome = CardRefresh(
            name=artefact.name, repo_id=artefact.repo_id, repo_type=repo_type, status="staged"
        )
        outcomes.append(outcome)
        try:
            card = stage_card(artefact, cfg)
            outcome.card = card.as_posix()
            if not dry_run:
                client = client or api or _api()
                client.upload_file(
                    path_or_fileobj=str(card),
                    path_in_repo=README_NAME,
                    repo_id=artefact.repo_id,
                    repo_type=repo_type,
                    commit_message="docs(card): regenerated from the current table and course",
                )
                outcome.status = "uploaded"
        except Exception as exc:  # noqa: BLE001 - one card failing must not stop the rest
            log.exception("card of %s failed", artefact.name)
            outcome.status = "failed"
            outcome.detail = f"{type(exc).__name__}: {exc}"
    return outcomes
