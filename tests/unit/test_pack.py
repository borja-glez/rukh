"""Tests for rukh.tokenize.pack: stream layout, index, metadata and the three encoders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from rukh.data.pipeline import TokenizeConfig
from rukh.tokenize.bpe import train_bpe_from_parquet
from rukh.tokenize.pack import (
    STARTS_FILE,
    TOKENS_FILE,
    BpeGameEncoder,
    GameSlice,
    SanGameEncoder,
    UciGameEncoder,
    make_encoder,
    pack_games,
    pack_month,
    read_pack_info,
    split_sources,
)
from rukh.tokenize.uci_vocab import UciTokenizer

TEST_BPE_VOCAB = 600
"""Small enough to train in a second, large enough to produce multi-move merges."""

pytestmark = pytest.mark.unit


@pytest.fixture
def games_parquet(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "games.parquet"


@pytest.fixture
def test_bpe(tmp_path: Path, games_parquet: Path) -> Tokenizer:
    """A BPE trained for the tests, never the one in ``artifacts/``.

    ``rukh data tokenize --scheme bpe`` legitimately rewrites ``artifacts/tokenizer/bpe.json``
    with the real 4 096-token vocabulary, and a test that pinned the committed file turned red
    the first time the pipeline ran for real. A test owns its fixtures.
    """
    out = tmp_path / "test-bpe.json"
    return train_bpe_from_parquet(games_parquet, out, vocab_size=TEST_BPE_VOCAB, n_games=20)


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


def test_pack_month_bytes_are_stable(
    tmp_path: Path, games_parquet: Path, test_bpe: Tokenizer
) -> None:
    """The packed bytes are a published contract: streaming must not change them."""
    expected = {
        "uci": (
            "a9df9aa63875adee99b42d4d1ef8ed2db7ba4dd087c5b0533dc88705bb82eb30",
            "91ffa8971d5552917893774c773245cfd980c0c7db1fe5c103140360cd08b1da",
        ),
        "bpe": (
            "028a4281a0cade6ccbb55abbe68a393354da394d74554caea12626c3afda5a18",
            "cc8eda0b46c069c982b6e9ad1a4ea1ac39dcf37f303a220a292e33919c72d60f",
        ),
    }
    encoders = {
        "uci": UciGameEncoder(),
        "bpe": BpeGameEncoder(test_bpe),
    }
    for scheme, encoder in encoders.items():
        out = tmp_path / scheme
        pack_month(games_parquet, encoder, out)
        digests = tuple(
            hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in (TOKENS_FILE, STARTS_FILE)
        )
        assert digests == expected[scheme], scheme


def test_san_and_bpe_encoders(tmp_path: Path, games_parquet: Path, test_bpe: Tokenizer) -> None:
    san = SanGameEncoder()
    ids = san.encode_game("e2e4 e7e5", 1800, 1800, "1-0", max_len=1000)
    assert ids[0] == 1 and ids[-1] == 2
    assert san.tokenizer.decode(ids) == "1.e4 e5 1-0"
    info = pack_month(games_parquet, san, tmp_path / "san")
    assert info.scheme == "san" and info.vocab_size == 35

    bpe = BpeGameEncoder(test_bpe)
    ids = bpe.encode_game("e2e4 e7e5", 1800, 1800, "0-1", max_len=1000)
    assert ids[0] == 1 and ids[-2] == 6 and ids[-1] == 2
    info = pack_month(games_parquet, bpe, tmp_path / "bpe")
    assert info.scheme == "bpe" and info.vocab_size == TEST_BPE_VOCAB
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


def test_slices_partition_a_parquet(tmp_path: Path, games_parquet: Path) -> None:
    """A head slice and its complement hold every game exactly once, in order.

    This is the invariant the ``val_games`` split rests on: the validation games and the games
    that join training are disjoint and together are the month, so nothing is trained on twice
    and nothing is silently dropped.
    """
    whole = pack_games([GameSlice(games_parquet)], UciGameEncoder(), tmp_path / "whole")
    cut = whole.n_games // 3
    head = pack_games([GameSlice(games_parquet, take=cut)], UciGameEncoder(), tmp_path / "head")
    tail = pack_games([GameSlice(games_parquet, skip=cut)], UciGameEncoder(), tmp_path / "tail")

    assert head.n_games == cut
    assert tail.n_games == whole.n_games - cut
    assert head.n_tokens + tail.n_tokens == whole.n_tokens

    expected = np.load(tmp_path / "whole" / TOKENS_FILE)
    joined = np.concatenate(
        [np.load(tmp_path / "head" / TOKENS_FILE), np.load(tmp_path / "tail" / TOKENS_FILE)]
    )
    assert np.array_equal(joined, expected)


def test_slice_smaller_than_a_chunk(tmp_path: Path, games_parquet: Path) -> None:
    """``take`` stops mid-batch: the packer must not round up to the row-group boundary."""
    info = pack_games([GameSlice(games_parquet, skip=1, take=2)], UciGameEncoder(), tmp_path / "p")
    assert info.n_games == 2
    starts = np.load(tmp_path / "p" / STARTS_FILE)
    assert starts[0] == 0


def test_concatenated_sources_index_absolute_offsets(tmp_path: Path, games_parquet: Path) -> None:
    """Two sources make one stream, and ``starts`` indexes the concatenation, not each source."""
    info = pack_games(
        [GameSlice(games_parquet), GameSlice(games_parquet)], UciGameEncoder(), tmp_path / "twice"
    )
    single = pack_games([GameSlice(games_parquet)], UciGameEncoder(), tmp_path / "once")
    assert info.n_games == 2 * single.n_games
    assert info.n_tokens == 2 * single.n_tokens

    tokens = np.load(tmp_path / "twice" / TOKENS_FILE)
    starts = np.load(tmp_path / "twice" / STARTS_FILE)
    assert starts[single.n_games] == single.n_tokens
    assert tokens[starts[single.n_games]] == UciGameEncoder().bos_id
    assert np.array_equal(tokens[: single.n_tokens], tokens[single.n_tokens :])


def test_slice_labels_reach_metadata(tmp_path: Path, games_parquet: Path) -> None:
    """``meta.json`` says which rows of which file it holds, so a pack is never ambiguous."""
    pack_games([GameSlice(games_parquet, skip=2, take=3)], UciGameEncoder(), tmp_path / "p")
    assert read_pack_info(tmp_path / "p").source.endswith("games.parquet[2:5]")


def test_val_games_splits_the_month_between_the_splits(tmp_path: Path) -> None:
    """``val_games`` sends the head of ``val_month`` to val and its tail to train."""
    cfg = TokenizeConfig(train_months=["2025-01"], val_month="2025-02", val_games=100)
    sources = split_sources(cfg)
    assert [s.take for s in sources["val"]] == [100]
    assert len(sources["train"]) == 2
    assert sources["train"][1].skip == 100 and sources["train"][1].take is None
    assert sources["train"][0].path != sources["val"][0].path
    assert sources["train"][1].path == sources["val"][0].path


def test_val_games_zero_keeps_the_whole_month_for_validation(tmp_path: Path) -> None:
    """The P1/P2 behaviour stays reachable, so old packs can be reproduced."""
    sources = split_sources(TokenizeConfig(train_months=["2025-01"], val_games=0))
    assert len(sources["train"]) == 1
    assert sources["val"][0].skip == 0 and sources["val"][0].take is None


def test_extra_parquets_join_training_only(tmp_path: Path, games_parquet: Path) -> None:
    """The Elite Database is appended to training and never leaks into validation."""
    cfg = TokenizeConfig(
        train_months=["2025-01"],
        val_month="2025-02",
        val_games=50,
        extra_train_parquets=[str(games_parquet)],
    )
    sources = split_sources(cfg)
    assert len(sources["val"]) == 1
    assert sources["val"][0].path.name == "games.parquet"
    assert sources["train"][-1].path == games_parquet
    assert games_parquet not in [s.path for s in sources["val"]]


def test_the_validation_remainder_can_be_kept_out_of_training(rukh_home: Path) -> None:
    """A corpus whose shape is the point must not be topped up with 2.85 M games of one band.

    The Elo-balanced sample of M4 has the same number of games per rating band on purpose: that
    is what teaches the model what `<w1500>` *means* rather than how rare it is. The default
    behaviour -- the rest of the validation month joins training -- would quietly undo it.
    """
    from rukh.data.pipeline import TokenizeConfig
    from rukh.tokenize.pack import split_sources

    for month in ("01", "02"):
        target = rukh_home / "data" / "uci" / "year=2025" / f"month={month}" / "games.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    shared = {"train_months": [], "val_month": "2025-02", "val_games": 100, "uci_dir": "data/uci"}
    default = split_sources(TokenizeConfig(**shared))  # type: ignore[arg-type]
    assert len(default["train"]) == 1  # the remainder of the month

    kept_out = split_sources(TokenizeConfig(**shared, val_remainder_trains=False))  # type: ignore[arg-type]
    assert kept_out["train"] == []
    assert kept_out["val"] == default["val"]  # the same held-out games either way
