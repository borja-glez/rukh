"""Publishing the reward model, which is not a policy and must not be published as one.

`rukh publish model` stages weights, three ONNX exports, a vocabulary and a card whose numbers come
from `rukh eval`. A reward model has none of that. It has no ONNX because nothing serves it in a
browser; it has no Elo because it does not play; and the suite that measures every other stage of
this project has no row it belongs in.

What it does have is a number that means something narrow and precise -- how often it orders two
moves the way an engine did -- and three caveats that have to travel with it or the number is
misleading:

* the scale is meaningless, because Bradley-Terry only ever sees differences;
* the headline accuracy depends on the **split**, since the seed decides how many mate pairs land
  in validation and that is the band the model is worst at;
* the two correlations it reports do not share a sign, and the reason is that same band.

So the card is written from the run's own `run.json` and leads with all three. A reward model
published with one accuracy and no band table would be publishing the easy half of what was
measured.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rukh.models.squares import SQUARE_TOKENS
from rukh.paths import resolve
from rukh.publish.links import course_links
from rukh.publish.model import CONFIG_NAME, README_NAME, ModelPublishConfig, _api, render_card

log = logging.getLogger(__name__)

__all__ = ["RewardPublishResult", "publish_reward"]

REPO_TYPE = "model"
REWARD_CARD_TEMPLATE = "reward.md.jinja"
SAFETENSORS_NAME = "model.safetensors"


class RewardPublishResult(BaseModel):
    """Where the folder was staged, what went in it, and whether it was uploaded."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str
    folder: str
    files: list[str]
    uploaded: bool
    accuracy: float
    card_path: str
    """The staged card. Every other publisher returns one, and `publish.cards` rebuilt this path
    by hand for the reward branch alone because this one did not."""


def _load(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(checkpoint payload, run result)`` for a reward run directory."""
    import torch

    checkpoint = run_dir / "reward.pt" if run_dir.is_dir() else run_dir
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result_path = checkpoint.parent / "run.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"{result_path} is missing; the card is written from it, so the run has to have "
            "been produced by `rukh train reward`"
        )
    return payload, json.loads(result_path.read_text(encoding="utf-8"))


def _seed_spread() -> str:
    """The sentence the card uses to say how little the headline deserves to be trusted.

    Hard-coded from the runs M5 actually did rather than computed -- which is why it takes no
    argument -- because the card is about *this* checkpoint while the spread is a property of the
    configuration. Two numbers matter and the second is the one nobody expects: four seeds spread
    over two and a half points, **and two runs of the same seed spread over 1.3** -- same split,
    same hyperparameters, different GPU kernels.
    """
    return (
        "72.91 %, 74.21 %, 74.71 %, 74.76 % and 75.19 %. The first two are the **same seed**, "
        "so 1.3 of those points are not the split: they are what this measurement repeats to"
    )


def publish_reward(
    run: Path,
    repo: str,
    cfg: ModelPublishConfig | None = None,
    stage: str | None = None,
    dry_run: bool = False,
) -> RewardPublishResult:
    """Stage (and unless ``dry_run``, upload) a reward model as its own Hub repository."""
    cfg = cfg or ModelPublishConfig()
    run = Path(run)
    payload, result = _load(run)
    repo_id = repo if "/" in repo else f"{cfg.owner}/{repo}"
    name = stage or repo_id.split("/")[-1].removeprefix("rukh-")

    encoder_config = payload["encoder_config"]
    state = payload["model"]
    params = sum(int(tensor.numel()) for tensor in state.values())

    folder = resolve(cfg.publish_dir) / repo_id
    folder.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    save_file(
        {key: value.detach().cpu().contiguous() for key, value in state.items()},
        str(folder / SAFETENSORS_NAME),
    )
    config = {
        "model_type": "rukh-reward",
        "stage": name,
        "params": params,
        "pooling": payload.get("pooling"),
        "encoder": encoder_config,
        "square_tokens": SQUARE_TOKENS,
        "objective": "bradley-terry",
        "training": result.get("checkpoint"),
    }
    (folder / CONFIG_NAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    training = payload.get("config") or {}
    card = render_card(
        {
            "license": cfg.license,
            "repo_id": repo_id,
            "stage": name,
            "repository_url": cfg.repository_url,
            "course_url": cfg.course_url,
            "course": course_links(repo_id, cfg.course_url, cfg.demo_url),
            "datasets": [f"{cfg.owner}/rukh-pairs-dpo"],
            "params": params,
            "n_layer": encoder_config.get("n_layer"),
            "d_model": encoder_config.get("d_model"),
            "square_tokens": SQUARE_TOKENS,
            "pairs": result["pairs"],
            "train_pairs": result["train_pairs"],
            "val_pairs": result["val_pairs"],
            "accuracy": result["accuracy"],
            "accuracy_without_mates": result.get("accuracy_without_mates"),
            "loss": result["loss"],
            "pearson": result.get("pearson"),
            "pearson_without_mates": result.get("pearson_without_mates"),
            "bands": result.get("bands") or [],
            "epochs": result.get("epochs"),
            "batch_size": training.get("batch_size"),
            "lr": training.get("lr"),
            "seed": training.get("seed"),
            "from_scratch": result.get("from_scratch", True),
            "seed_spread": _seed_spread(),
        },
        REWARD_CARD_TEMPLATE,
    )
    card_path = folder / README_NAME
    card_path.write_text(card, encoding="utf-8", newline="\n")

    files = [SAFETENSORS_NAME, CONFIG_NAME, README_NAME]
    uploaded = False
    if not dry_run:
        api = _api()
        api.create_repo(repo_id, repo_type=REPO_TYPE, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(folder), repo_type=REPO_TYPE)
        uploaded = True
        log.info("uploaded %s", repo_id)

    return RewardPublishResult(
        repo_id=repo_id,
        folder=str(folder),
        files=files,
        uploaded=uploaded,
        accuracy=float(result["accuracy"]),
        card_path=str(card_path),
    )
