"""Tests for rukh.tokenize.pack: stream layout, index, metadata and the three encoders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rukh.tokenize.bpe import load_bpe
from rukh.tokenize.pack import (
    BpeGameEncoder,
    SanGameEncoder,
    UciGameEncoder,
    make_encoder,
    pack_month,
    read_pack_info,
)
from rukh.tokenize.uci_vocab import UciTokenizer

pytestmark = pytest.mark.unit


@pytest.fixture
def games_parquet(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "games.parquet"


def test_pack_month_layout(tmp_path: Path, games_parquet: Path) -> None:
    info = pack_month(games_parquet, UciGameEncoder(), tmp_path / "pack")
    tokens = np.load(tmp_path / "pack" / "tokens.npy", mmap_mode="r")
    starts = np.load(tmp_path / "pack" / "starts.npy")
    assert tokens.dtype == np.uint16 and starts.dtype == np.int64
    assert info.n_games == 20 and len(starts) == 20
    assert info.n_tokens == len(tokens)
    assert tokens[-1] == UciTokenizer.eos_id
    assert all(tokens[s] == UciTokenizer.bos_id for s in starts)
    assert starts[0] == 0 and np.all(np.diff(starts) > 0)
    # Every game is stored whole: <bos>, two Elo tokens, plies, result, <eos>.
    import polars as pl

    plies = pl.read_parquet(games_parquet.as_posix())["n_plies"].to_list()
    lengths = np.diff(np.append(starts, len(tokens)))
    assert lengths.tolist() == [p + 5 for p in plies]
    assert not (tmp_path / "pack" / "tokens.bin").exists()
    meta = json.loads((tmp_path / "pack" / "meta.json").read_text("utf-8"))
    assert meta["scheme"] == "uci" and meta["vocab_size"] == 2030
    assert len(meta["vocab_hash"]) == 64
    assert read_pack_info(tmp_path / "pack") == info


def test_san_and_bpe_encoders(tmp_path: Path, games_parquet: Path, repo_root: Path) -> None:
    san = SanGameEncoder()
    ids = san.encode_game("e2e4 e7e5", 1800, 1800, "1-0", max_len=1000)
    assert ids[0] == 1 and ids[-1] == 2
    assert san.tokenizer.decode(ids) == "1.e4 e5 1-0"
    info = pack_month(games_parquet, san, tmp_path / "san")
    assert info.scheme == "san" and info.vocab_size == 35

    bpe = BpeGameEncoder(load_bpe(repo_root / "artifacts" / "tokenizer" / "bpe.json"))
    ids = bpe.encode_game("e2e4 e7e5", 1800, 1800, "0-1", max_len=1000)
    assert ids[0] == 1 and ids[-2] == 6 and ids[-1] == 2
    info = pack_month(games_parquet, bpe, tmp_path / "bpe")
    assert info.scheme == "bpe" and info.vocab_size == 600
    tokens = np.load(tmp_path / "bpe" / "tokens.npy")
    assert tokens[-1] == 2 and tokens[0] == 1


def test_make_encoder() -> None:
    assert make_encoder("uci").scheme == "uci"
    assert make_encoder("san").scheme == "san"
    with pytest.raises(ValueError):
        make_encoder("bpe")
    with pytest.raises(ValueError):
        make_encoder("nope")


def test_cli_pack_writes_train_and_val(rukh_home: Path, games_parquet: Path) -> None:
    import shutil

    from typer.testing import CliRunner

    from rukh.cli import app

    for month in ("01", "02"):
        target = rukh_home / "data" / "uci" / "year=2025" / f"month={month}" / "games.parquet"
        target.parent.mkdir(parents=True)
        shutil.copy(games_parquet, target)
    result = CliRunner().invoke(app, ["data", "tokenize", "--scheme", "uci", "--pack"])
    assert result.exit_code == 0, result.output
    for split in ("train", "val"):
        info = read_pack_info(rukh_home / "data" / "tokens" / "uci" / split)
        assert info.n_games == 20
