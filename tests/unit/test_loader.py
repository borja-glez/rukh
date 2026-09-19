"""Tests for rukh.tokenize.loader: <bos>-aligned windows, padding, determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rukh.tokenize.loader import IGNORE_INDEX, PackedDataset, make_loader
from rukh.tokenize.pack import UciGameEncoder, pack_month
from rukh.tokenize.uci_vocab import UciTokenizer

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = Path(__file__).resolve().parents[2]
    out = tmp_path_factory.mktemp("pack")
    pack_month(root / "tests" / "fixtures" / "games.parquet", UciGameEncoder(), out)
    return out


def test_items_are_bos_aligned_windows(pack_dir: Path) -> None:
    ds = PackedDataset(pack_dir, block=200)
    assert len(ds) == 20
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (200,) and y.shape == (200,)
        assert x.dtype == torch.int64
        assert x[0] == UciTokenizer.bos_id
        assert torch.equal(y[:-1], x[1:])
        assert y[-1] == IGNORE_INDEX


def test_last_window_is_padded(pack_dir: Path) -> None:
    ds = PackedDataset(pack_dir, block=400)
    x, y = ds[len(ds) - 1]
    tokens = np.load(pack_dir / "tokens.npy")
    tail = len(tokens) - int(ds.starts[-1])
    assert tail < 400
    assert x[tail - 1] == UciTokenizer.eos_id
    assert torch.all(x[tail:] == IGNORE_INDEX)
    assert torch.all(y[tail - 1 :] == IGNORE_INDEX)


def test_contiguous_blocks(pack_dir: Path) -> None:
    ds = PackedDataset(pack_dir, block=64, start_at_game=False)
    tokens = np.load(pack_dir / "tokens.npy")
    assert len(ds) == len(tokens) // 64
    x, _ = ds[1]
    assert torch.equal(x, torch.from_numpy(tokens[64:128].astype(np.int64)))
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_same_seed_same_batches(pack_dir: Path) -> None:
    ds = PackedDataset(pack_dir, block=200)
    a = make_loader(ds, batch_size=4, seed=7)
    b = make_loader(ds, batch_size=4, seed=7)
    c = make_loader(ds, batch_size=4, seed=8)
    batches_a = [x for x, _ in a]
    batches_b = [x for x, _ in b]
    batches_c = [x for x, _ in c]
    assert len(batches_a) == 5
    assert all(torch.equal(p, q) for p, q in zip(batches_a, batches_b, strict=True))
    assert any(not torch.equal(p, q) for p, q in zip(batches_a, batches_c, strict=True))
    assert batches_a[0].shape == (4, 200)


def test_thousand_samples_in_range(pack_dir: Path) -> None:
    ds = PackedDataset(pack_dir, block=200)
    rng = np.random.default_rng(0)
    for i in rng.integers(0, len(ds), size=1000):
        x, y = ds[int(i)]
        assert int(x.max()) < 2030 and int(y.max()) < 2030
        assert int(x.min()) >= 0 and int(y.min()) >= 0


def test_dataset_pickles_without_carrying_its_arrays(pack_dir: Path) -> None:
    """A pickled dataset ships paths, not data, and still reads the same windows.

    DataLoader workers are spawned on Windows, so the dataset crosses a pickle once per worker.
    `starts` is one int64 per game, which reached 145 MB at 18.9 M games and killed the spawn
    with `UnpicklingError: pickle data was truncated`. The arrays must be reopened in the worker.
    """
    import pickle

    dataset = PackedDataset(pack_dir, block=32)

    state = dataset.__getstate__()
    assert "tokens" not in state and "starts" not in state

    revived = pickle.loads(pickle.dumps(dataset))
    assert len(revived) == len(dataset)
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        x, y = dataset[index]
        rx, ry = revived[index]
        assert torch.equal(x, rx) and torch.equal(y, ry)
