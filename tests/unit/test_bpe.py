"""Tests for rukh.tokenize.bpe: training, exact round trip, saved file and statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from typer.testing import CliRunner

from rukh.cli import app
from rukh.tokenize.bpe import (
    bpe_text,
    decode,
    encode,
    load_bpe,
    longest_tokens,
    save,
    split_moves,
    train_bpe,
)
from rukh.tokenize.fixture import read_pgn_games
from rukh.tokenize.uci_vocab import UciTokenizer

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def games(repo_root: Path) -> list[str]:
    return [str(g["uci"]) for g in read_pgn_games(repo_root / "tests" / "fixtures" / "games.pgn")]


@pytest.fixture(scope="module")
def trained(games: list[str]) -> Tokenizer:
    return train_bpe((bpe_text(g) for g in games), vocab_size=600)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_specials_and_config(trained: Tokenizer) -> None:
    assert trained.get_vocab_size() == 600
    for i, token in enumerate(UciTokenizer.specials):
        assert trained.token_to_id(token) == i
    payload = json.loads(trained.to_str())
    assert payload["model"]["type"] == "BPE"
    assert payload["model"]["unk_token"] == "<unk>"
    assert payload["model"]["continuing_subword_prefix"] is None
    assert payload["model"]["end_of_word_suffix"] is None
    assert payload["pre_tokenizer"]["type"] == "WhitespaceSplit"
    assert payload["normalizer"] is None
    assert payload["decoder"] is None


def test_round_trip_is_exact(trained: Tokenizer, games: list[str]) -> None:
    for uci in games:
        ids = encode(trained, bpe_text(uci))
        assert decode(trained, ids) == uci
        assert UciTokenizer.unk_id not in ids


def test_learns_multi_move_tokens(trained: Tokenizer) -> None:
    longest = longest_tokens(trained, 10)
    assert len(longest) == 10
    assert longest[0]["moves"] >= 2
    assert longest[0]["length"] == len(longest[0]["token"])


def test_save_and_load(tmp_path: Path, trained: Tokenizer, games: list[str]) -> None:
    path = tmp_path / "bpe.json"
    save(trained, path)
    again = Tokenizer.from_file(str(path))
    assert again.encode(bpe_text(games[3])).ids == trained.encode(bpe_text(games[3])).ids
    assert load_bpe(path).get_vocab_size() == 600


def test_split_moves_handles_promotion_ambiguity() -> None:
    assert split_moves("e2e4e7e5") == ["e2e4", "e7e5"]
    # 4...Bxc1 is a bishop move from rank 2 to rank 1 followed by 5.b4: the 'b' is a file.
    bishop = "d2d3 e7e6 e1d2 f8b4 d2e3 b4d2 e3f3 d2c1 b2b4"
    assert split_moves(bishop.replace(" ", "")) == bishop.split()
    # 5.gxh8=B then 5...Nc6: the first 'b' is the promotion piece, the second a file.
    promo = "h2h4 g7g5 h4g5 h7h6 g5g6 f7f6 g6g7 f6f5 g7h8b b8c6"
    assert split_moves(promo.replace(" ", "")) == promo.split()
    # After an illegal move the split falls back to the rank rule.
    assert split_moves("e2e5e7e8qa1a2") == ["e2e5", "e7e8q", "a1a2"]


def test_committed_bpe_matches_fixture(repo_root: Path) -> None:
    bpe = load_bpe(repo_root / "artifacts" / "tokenizer" / "bpe.json")
    fixture = json.loads(
        (repo_root / "artifacts" / "tokenizer" / "fixtures" / "games.json").read_text("utf-8")
    )
    for entry in fixture:
        assert entry["bpe_ids"] == encode(bpe, bpe_text(entry["uci"])), entry["id"]
        assert decode(bpe, entry["bpe_ids"]) == entry["uci"]


def test_stats_lengths_match_the_encoders(repo_root: Path, trained: Tokenizer) -> None:
    """The statistics count exactly what each encoder emits (BPE frames with three tokens)."""
    import polars as pl

    from rukh.data.pipeline import TokenizeConfig
    from rukh.tokenize.pack import BpeGameEncoder, SanGameEncoder, UciGameEncoder
    from rukh.tokenize.run import compute_stats

    games = repo_root / "tests" / "fixtures" / "games.parquet"
    stats = compute_stats(TokenizeConfig(), games, trained)
    frame = pl.read_parquet(games.as_posix()).select("uci", "white_elo", "black_elo", "result")
    encoders = {
        "uci": UciGameEncoder(),
        "san": SanGameEncoder(),
        "bpe": BpeGameEncoder(trained),
    }
    for scheme, encoder in encoders.items():
        lengths = [
            len(encoder.encode_game(uci, int(w), int(b), result, max_len=1 << 30))
            for uci, w, b, result in frame.iter_rows()
        ]
        assert stats["schemes"][scheme]["mean"] == round(sum(lengths) / len(lengths), 2), scheme


def test_cli_stats_and_bpe_training(rukh_home: Path, repo_root: Path) -> None:
    games = (repo_root / "tests" / "fixtures" / "games.parquet").as_posix()
    pgn = (repo_root / "tests" / "fixtures" / "games.pgn").as_posix()
    cfg = rukh_home / "pipeline.yaml"
    cfg.write_text(
        f"tokenize:\n  fixture_pgn: {pgn}\n  bpe_vocab_size: 300\n  bpe_train_games: 20\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        [
            "data",
            "tokenize",
            "--config",
            str(cfg),
            "--scheme",
            "bpe",
            "--stats",
            "--export-fixture",
            "--games",
            games,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "vocab_size: 300" in result.output
    bpe_path = rukh_home / "artifacts" / "tokenizer" / "bpe.json"
    assert Tokenizer.from_file(str(bpe_path)).get_vocab_size() == 300
    stats = json.loads((rukh_home / "artifacts" / "web" / "tokenizer-stats.json").read_text())
    assert set(stats["schemes"]) == {"uci", "san", "bpe"}
    assert stats["schemes"]["uci"]["vocab_size"] == 2030
    assert stats["schemes"]["san"]["vocab_size"] == 35
    assert stats["schemes"]["bpe"]["vocab_size"] == 300
    assert stats["schemes"]["san"]["mean"] > stats["schemes"]["uci"]["mean"]
    assert stats["source"]["n_games"] == 20
    assert len(stats["bpe_longest_tokens"]) == 10
    fixture = json.loads(
        (rukh_home / "artifacts" / "tokenizer" / "fixtures" / "games.json").read_text("utf-8")
    )
    assert all("bpe_ids" in e and "san_ids" in e for e in fixture)
