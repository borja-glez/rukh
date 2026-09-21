"""Tests for rukh.train.checkpoint: round-trip, provenance and RNG restoration."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from rukh.models import DecoderConfig, MoveDecoder
from rukh.train import (
    TrainConfig,
    load_checkpoint,
    read_manifest_sha,
    read_vocab_hash,
    restore,
    save_checkpoint,
    step_name,
)

pytestmark = pytest.mark.unit

TOY = DecoderConfig(vocab_size=32, n_layer=2, n_head=2, d_model=32, block=16)


def make_pair() -> tuple[MoveDecoder, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = MoveDecoder(TOY)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def test_save_and_load_reproduce_weights_and_step(tmp_path: Path) -> None:
    model, optimizer = make_pair()
    idx = torch.randint(1, TOY.vocab_size, (2, 8))
    _, loss = model(idx, idx)
    assert loss is not None
    loss.backward()
    optimizer.step()

    path = save_checkpoint(
        tmp_path / step_name(7),
        step=7,
        model=model,
        optimizer=optimizer,
        cfg=TrainConfig(max_steps=7).model_dump(mode="json"),
        model_cfg=TOY.model_dump(mode="json"),
        vocab_hash="abc",
        data_manifest_sha=None,
        git_sha="deadbeef",
        best_val=1.25,
    )
    assert path.name == "step-7.pt"
    assert not list(tmp_path.glob("*.tmp"))

    payload = load_checkpoint(path)
    assert payload["step"] == 7
    assert payload["vocab_hash"] == "abc"
    assert payload["data_manifest_sha"] is None
    assert payload["git_sha"] == "deadbeef"
    assert payload["best_val"] == pytest.approx(1.25)
    assert payload["cfg"]["max_steps"] == 7
    assert payload["model_cfg"]["d_model"] == 32

    fresh = MoveDecoder(TOY)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    assert restore(payload, fresh, fresh_opt) == 7
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    with torch.no_grad():
        assert torch.allclose(model.eval()(idx)[0], fresh.eval()(idx)[0], atol=1e-6)


def test_restoring_the_rng_repeats_the_same_draws(tmp_path: Path) -> None:
    model, optimizer = make_pair()
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    path = save_checkpoint(
        tmp_path / "rng.pt",
        step=0,
        model=model,
        optimizer=optimizer,
        cfg={},
        model_cfg=TOY.model_dump(mode="json"),
    )
    expected = (random.random(), float(np.random.random()), torch.rand(3).tolist())

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore(load_checkpoint(path), MoveDecoder(TOY))
    assert (random.random(), float(np.random.random()), torch.rand(3).tolist()) == expected


def test_vocab_hash_and_manifest_sha_are_defensive(tmp_path: Path) -> None:
    assert read_vocab_hash(tmp_path) is None
    (tmp_path / "meta.json").write_text(json.dumps({"n_games": 1}), encoding="utf-8")
    assert read_vocab_hash(tmp_path) is None
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_hash": "cafe"}), encoding="utf-8")
    assert read_vocab_hash(tmp_path) == "cafe"

    assert read_manifest_sha(tmp_path / "does-not-exist" / "manifest.json") is None
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{}")
    sha = read_manifest_sha(manifest)
    assert sha is not None and len(sha) == 64


def test_a_stable_run_name_resolves_to_the_newest_stamped_run(tmp_path: Path) -> None:
    """``checkpoints/small/best.pt`` finds ``checkpoints/small-<newest stamp>/best.pt``.

    This is what lets a config or a lesson name a run without its date: the reader who trained
    it has the stamped folder, the reader who pulled it from the Hub has the stable one, and the
    same spelling serves both.
    """
    from rukh.train.checkpoint import resolve_run

    root = tmp_path / "checkpoints"
    older = root / "small-20260919-062911"
    newer = root / "small-20260920-150000"
    decoy = root / "small-v3-20260921-000000"  # another run whose name merely starts the same
    for folder in (older, newer, decoy):
        folder.mkdir(parents=True)
        (folder / "best.pt").write_bytes(b"x")
    (root / "small-20260921-000000").mkdir()  # newest stamp, but it holds no best.pt yet

    assert resolve_run(root / "small" / "best.pt") == newer / "best.pt"
    assert resolve_run(root / "small") == root / "small-20260921-000000"
    assert resolve_run(root / "small-v3" / "best.pt") == decoy / "best.pt"
    # A stable folder wins over any stamped one, and an unknown name comes back untouched.
    (root / "small").mkdir()
    (root / "small" / "best.pt").write_bytes(b"y")
    assert resolve_run(root / "small" / "best.pt") == root / "small" / "best.pt"
    assert resolve_run(root / "medium" / "best.pt") == root / "medium" / "best.pt"
    assert resolve_run(tmp_path / "nowhere" / "run" / "best.pt") == (
        tmp_path / "nowhere" / "run" / "best.pt"
    )


def test_load_checkpoint_reads_through_the_stable_name(tmp_path: Path) -> None:
    model, optimizer = make_pair()
    run = tmp_path / "checkpoints" / "tiny-20260919-060101"
    save_checkpoint(
        run / "best.pt",
        step=5,
        model=model,
        optimizer=optimizer,
        cfg={},
        model_cfg=TOY.model_dump(),
    )
    payload = load_checkpoint(tmp_path / "checkpoints" / "tiny" / "best.pt")
    assert payload["step"] == 5


def test_a_foreign_file_is_not_a_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(2)}, path)
    with pytest.raises(ValueError, match="not a rukh checkpoint"):
        load_checkpoint(path)
