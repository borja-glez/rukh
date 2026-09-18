"""Tests for rukh.tokenize.uci_vocab: enumeration, encode/decode, export and the fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rukh.cli import app
from rukh.tokenize.fixture import build_fixture, read_pgn_games
from rukh.tokenize.uci_vocab import (
    SPECIALS,
    UciTokenizer,
    build_moves,
    build_promotions,
    build_vocab,
    elo_bin,
    elo_token,
    squares,
)

pytestmark = pytest.mark.unit

VOCAB_SHA256 = "f6aa692bbd581c124dda4db0a088111580df4d311974bb5d4ec29b4d761142f1"


@pytest.fixture(scope="module")
def tok() -> UciTokenizer:
    return UciTokenizer()


def test_squares_are_file_major() -> None:
    sq = squares()
    assert sq[:9] == ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "b1"]
    assert sq[-1] == "h8"
    assert sq.index("c5") == 2 * 8 + 4


def test_enumeration_sizes(tok: UciTokenizer) -> None:
    assert len(build_moves()) == 1792
    assert len(build_promotions()) == 176
    assert len(build_vocab()) == 2030
    assert len(tok) == 2030
    assert len(tok.vocab) == 2030


def test_special_and_elo_ids(tok: UciTokenizer) -> None:
    assert tok.ids[:8] == SPECIALS
    assert tok.vocab["<pad>"] == 0 and tok.vocab["<bos>"] == 1 and tok.vocab["<eos>"] == 2
    assert tok.vocab["<mask>"] == 3 and tok.vocab["<unk>"] == 4
    assert tok.vocab["<1-0>"] == 5 and tok.vocab["<0-1>"] == 6 and tok.vocab["<1/2>"] == 7
    assert tok.vocab["<w0600>"] == 8 and tok.vocab["<w3200>"] == 34
    assert tok.vocab["<b0600>"] == 35 and tok.vocab["<b3200>"] == 61
    assert tok.ids[62] == "a1a2"
    assert tok.ids[-1] == "h2h1n"


def test_moves_and_promotions_present(tok: UciTokenizer) -> None:
    for move in ("e2e4", "e7e8q", "g1f3", "a1a8", "a1h8", "e1g1", "e1c1", "a1b3"):
        assert move in tok.vocab, move
    # Neither a queen nor a knight reaches these.
    for move in ("a1b4", "a1c4", "e2e2", "e7e8k", "e6e7q", "a7c8q"):
        assert move not in tok.vocab, move


def test_promotion_order(tok: UciTokenizer) -> None:
    promos = build_promotions()
    assert promos[:8] == ["a7a8q", "a7a8r", "a7a8b", "a7a8n", "a7b8q", "a7b8r", "a7b8b", "a7b8n"]
    assert promos[88] == "a2a1q"
    assert tok.vocab["a7a8q"] == 62 + 1792


@pytest.mark.parametrize(
    ("elo", "expected"),
    [
        (0, 600),
        (599, 600),
        (600, 600),
        (1850, 1800),
        (1899, 1800),
        (3199, 3100),
        (3200, 3200),
        (3299, 3200),
        (3300, 3200),
        (9999, 3200),
    ],
)
def test_elo_bin_clamps(elo: int, expected: int) -> None:
    assert elo_bin(elo) == expected


def test_elo_token_format() -> None:
    assert elo_token(650, "w") == "<w0600>"
    assert elo_token(2412, "b") == "<b2400>"
    with pytest.raises(ValueError):
        elo_token(1800, "x")


def test_encode_game_layout(tok: UciTokenizer) -> None:
    ids = tok.encode_game("e2e4 e7e5", 1850, 1920, "1-0")
    assert ids == [
        1,
        tok.vocab["<w1800>"],
        tok.vocab["<b1900>"],
        tok.vocab["e2e4"],
        tok.vocab["e7e5"],
        5,
        2,
    ]
    assert tok.encode_game("", 1000, 1000, "1/2-1/2")[-2:] == [7, 2]
    assert tok.encode_game("", 1000, 1000, "0-1")[-2:] == [6, 2]


def test_encode_game_truncates(tok: UciTokenizer) -> None:
    moves = " ".join(["g1f3", "g8f6", "f3g1", "f6g8"] * 60)
    ids = tok.encode_game(moves, 1800, 1800, "1/2-1/2", max_len=200)
    assert len(ids) == 200
    assert ids[-1] != tok.eos_id
    full = tok.encode_game(moves, 1800, 1800, "1/2-1/2", max_len=10_000)
    assert len(full) == 3 + 240 + 2 and full[-1] == tok.eos_id


def test_encode_game_unknown_move_and_result(tok: UciTokenizer) -> None:
    assert tok.encode_game("zz99", 1800, 1800, "1-0")[3] == tok.unk_id
    with pytest.raises(ValueError):
        tok.encode_game("e2e4", 1800, 1800, "*")


def test_decode_roundtrip(tok: UciTokenizer) -> None:
    ids = tok.encode_game("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5c6 d7c6", 2100, 2050, "0-1")
    tokens = tok.decode(ids)
    assert tokens == [
        "<bos>",
        "<w2100>",
        "<b2000>",
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1b5",
        "a7a6",
        "b5c6",
        "d7c6",
        "<0-1>",
        "<eos>",
    ]
    assert [tok.vocab[t] for t in tokens] == ids


def test_export_is_stable(tmp_path: Path, tok: UciTokenizer) -> None:
    path = tmp_path / "vocab.json"
    tok.export(path)
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == VOCAB_SHA256
    payload = json.loads(data)
    assert payload["version"] == 1
    assert payload["size"] == 2030
    assert payload["specials"] == SPECIALS
    assert payload["elo_bins"] == {"min": 600, "max": 3200, "step": 100}
    assert payload["tokens"] == tok.ids
    assert UciTokenizer.from_file(path).ids == tok.ids


def test_committed_vocab_matches(repo_root: Path) -> None:
    path = repo_root / "artifacts" / "tokenizer" / "vocab.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == VOCAB_SHA256


def test_fixture_pgn_has_20_legal_varied_games(repo_root: Path) -> None:
    games = read_pgn_games(repo_root / "tests" / "fixtures" / "games.pgn")
    assert len(games) == 20
    results = {g["result"] for g in games}
    assert results == {"1-0", "0-1", "1/2-1/2"}
    plies = [len(str(g["uci"]).split()) for g in games]
    assert min(plies) < 10 and max(plies) > 200
    all_uci = " ".join(str(g["uci"]) for g in games).split()
    assert "e1g1" in all_uci and "e1c1" in all_uci and "e8g8" in all_uci
    assert any(m.endswith("n") and len(m) == 5 for m in all_uci)


def test_committed_fixture_matches_encoder(repo_root: Path, tok: UciTokenizer) -> None:
    fixture = json.loads(
        (repo_root / "artifacts" / "tokenizer" / "fixtures" / "games.json").read_text("utf-8")
    )
    entries = build_fixture(repo_root / "tests" / "fixtures" / "games.pgn", tok)
    assert len(fixture) == 20
    for got, want in zip(fixture, entries, strict=True):
        for key in ("id", "white_elo", "black_elo", "result", "uci", "san", "uci_ids"):
            assert got[key] == want[key], (got["id"], key)
        assert len(got["uci_ids"]) <= 200
        assert got["uci_ids"][0] == 1


def test_cli_tokenize_exports_fixture(rukh_home: Path, repo_root: Path) -> None:
    cfg = rukh_home / "pipeline.yaml"
    pgn = (repo_root / "tests" / "fixtures" / "games.pgn").as_posix()
    cfg.write_text(f"tokenize:\n  fixture_pgn: {pgn}\n", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["data", "tokenize", "--config", str(cfg), "--scheme", "uci", "--export-fixture"]
    )
    assert result.exit_code == 0, result.output
    assert (rukh_home / "artifacts" / "tokenizer" / "vocab.json").is_file()
    assert (rukh_home / "artifacts" / "tokenizer" / "fixtures" / "games.json").is_file()
    assert "vocab_size: 2030" in result.output


def test_cli_tokenize_rejects_bad_scheme() -> None:
    result = CliRunner().invoke(app, ["data", "tokenize", "--scheme", "nope"])
    assert result.exit_code == 2
