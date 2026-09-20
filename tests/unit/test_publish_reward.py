"""Publishing the reward model: the card has to carry the caveats, not just the number.

An accuracy on its own is the easy half of what was measured. These tests pin the other half --
the band table, the two correlations and the note about the split -- because that is exactly what
a card generated from a template quietly loses when somebody edits the template.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rukh.models.config import EncoderConfig
from rukh.models.encoder import PositionEncoder
from rukh.models.reward import RewardModel
from rukh.publish.reward import publish_reward

pytestmark = pytest.mark.unit

TOY = EncoderConfig(input="squares", n_layer=1, n_head=2, d_model=16, dropout=0.0)

RUN = {
    "checkpoint": "checkpoints/rm-toy/reward.pt",
    "pairs": 13838,
    "train_pairs": 12376,
    "val_pairs": 1462,
    "accuracy": 0.7291,
    "accuracy_without_mates": 0.7362,
    "loss": 0.5244,
    "pearson": -0.034,
    "spearman": -0.012,
    "pearson_without_mates": 0.058,
    "spearman_without_mates": 0.075,
    "epochs": 10,
    "from_scratch": True,
    "point_of_view": "white",
    "bands": [
        {"band": "100-200", "pairs": 616.0, "accuracy": 0.7419},
        {"band": "mate", "pairs": 484.0, "accuracy": 0.7149},
    ],
}


def _run_dir(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = RewardModel(PositionEncoder(TOY))
    folder = tmp_path / "rm-toy"
    folder.mkdir()
    torch.save(
        {
            "model": model.state_dict(),
            "encoder_config": TOY.model_dump(mode="json"),
            "pooling": "mean",
            "config": {"batch_size": 64, "lr": 0.0003, "seed": 42},
        },
        folder / "reward.pt",
    )
    (folder / "run.json").write_text(json.dumps(RUN), encoding="utf-8")
    return folder


def test_the_card_carries_the_bands_and_both_correlations(rukh_home, tmp_path):
    result = publish_reward(_run_dir(tmp_path), "chorcat/rukh-rm", dry_run=True)
    assert result.uploaded is False
    # The env var is RUKH_HOME, not RUKH_ROOT. Getting it wrong is silent -- the test passes and
    # stages into the real `artifacts/publish/`, quietly replacing a published folder with toy
    # weights. It happened once; this line is why it cannot happen twice.
    assert Path(result.folder).is_relative_to(rukh_home)
    card = (Path(result.folder) / "README.md").read_text(encoding="utf-8")

    # The number, and the number that is not the same number.
    assert "72.91 %" in card
    assert "73.62 %" in card
    # Both correlations, with their signs, because they do not share one.
    assert "-0.034" in card
    assert "+0.058" in card
    # The band table, including the one the model is worst at.
    assert "| mate | 484 | 71.49 % |" in card
    # And the sentence that stops the headline being read as stable.
    assert "75.19 %" in card, "the spread across seeds has to travel with the number"
    assert "same seed" in card, "and so does the part of it that is not the split"


def test_the_run_has_to_have_been_measured(rukh_home, tmp_path):
    """A card written from a missing file would be a card of defaults."""
    folder = _run_dir(tmp_path)
    (folder / "run.json").unlink()
    with pytest.raises(FileNotFoundError):
        publish_reward(folder, "chorcat/rukh-rm", dry_run=True)


def test_nothing_reaches_the_network_on_a_dry_run(rukh_home, tmp_path, monkeypatch):
    def _boom():
        raise AssertionError("a dry run must not build an API client")

    monkeypatch.setattr("rukh.publish.reward._api", _boom)
    result = publish_reward(_run_dir(tmp_path), "chorcat/rukh-rm", dry_run=True)
    assert set(result.files) == {"model.safetensors", "config.json", "README.md"}
