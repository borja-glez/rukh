"""Tests for rukh.data.pgn_text: the same games, spelled the way a general LLM reads them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import chess
import polars as pl
import pytest

from rukh.data.pgn_text import PgnTextConfig, build, game_text, movetext, prompt_for

pytestmark = pytest.mark.unit

RUY_LOPEZ = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]


def _board(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board


def test_movetext_is_numbered_san() -> None:
    assert movetext(_board(RUY_LOPEZ)) == "1. e4 e5 2. Nf3 Nc6 3. Bb5"


def test_san_is_disambiguated_by_the_position_it_is_written_in() -> None:
    """The whole reason the movetext is replayed instead of read off the move stack.

    With both knights able to reach the square, the same move is `Nbd2`, not `Nd2`; only the
    position at that moment knows which letter is needed.
    """
    board = _board(["g1f3", "d7d5", "d2d4", "g8f6", "b1d2"])
    assert movetext(board).endswith("Nbd2")


def test_a_game_carries_its_ratings_and_its_result() -> None:
    text = game_text(" ".join(RUY_LOPEZ), 1800, 1900, "1-0", 120)
    assert text is not None
    assert text.startswith('[WhiteElo "1800"] [BlackElo "1900"]\n')
    assert text.endswith(" 1-0")
    assert "1. e4 e5" in text


def test_a_game_longer_than_the_cap_is_cut_not_dropped() -> None:
    text = game_text(" ".join(RUY_LOPEZ), 1800, 1800, "1-0", 4)
    assert text is not None
    assert text.rstrip(" 1-0").endswith("Nc6")


def test_an_unplayable_game_is_dropped_rather_than_repaired() -> None:
    assert game_text("e2e4 e2e4", 1800, 1800, "1-0", 120) is None
    assert game_text("", 1800, 1800, "1-0", 120) is None


def test_the_prompt_never_ends_on_a_space() -> None:
    """A BPE writes a move as " Nc6" with the space attached.

    A prompt that already carries the space forces the model onto the spelling it almost never
    saw in training, which would show up as a worse model rather than as a worse prompt.
    """
    for plies in range(len(RUY_LOPEZ) + 1):
        prompt = prompt_for(_board(RUY_LOPEZ[:plies]))
        assert prompt == prompt.rstrip()


def test_the_prompt_ends_with_the_move_number_when_white_is_to_move() -> None:
    assert prompt_for(_board(RUY_LOPEZ[:4])).endswith("Nc6 3.")


def test_the_prompt_ends_after_the_white_move_when_black_is_to_move() -> None:
    assert prompt_for(_board(RUY_LOPEZ)).endswith("3. Bb5")


def test_a_position_from_a_fen_carries_the_fen_and_the_ellipsis() -> None:
    """A puzzle has no history; asking the model to invent one is asking a different question."""
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 5 4"
    prompt = prompt_for(chess.Board(fen), 2000, 2000)
    assert f'[FEN "{fen}"]' in prompt
    assert prompt.endswith("4...")


def test_the_start_position_carries_no_fen_tag() -> None:
    assert "FEN" not in prompt_for(chess.Board())


def test_the_ratings_asked_for_are_the_ratings_written() -> None:
    assert '[WhiteElo "1200"] [BlackElo "2400"]' in prompt_for(chess.Board(), 1200, 2400)


def test_build_writes_both_splits_and_counts_what_it_dropped(
    rukh_home: Path, repo_root: Path
) -> None:
    src = repo_root / "tests" / "fixtures" / "games.parquet"
    for month in ("01", "02"):
        target = rukh_home / "data" / "uci" / "year=2025" / f"month={month}" / "games.parquet"
        target.parent.mkdir(parents=True)
        shutil.copy(src, target)
    rows = pl.read_parquet(src.as_posix()).height

    manifest = build(PgnTextConfig(n_train=rows, n_val=3))
    out = rukh_home / "data" / "pgn-text"
    train = [json.loads(line) for line in (out / "train.jsonl").read_text("utf-8").splitlines()]
    val = [json.loads(line) for line in (out / "val.jsonl").read_text("utf-8").splitlines()]
    assert manifest.counts == {"train": len(train), "val": len(val)}
    assert len(val) <= 3
    assert all(set(sample) == {"text"} for sample in train)
    assert all(sample["text"].startswith("[WhiteElo ") for sample in train)
    assert manifest.filters["dropped"] == {"train": 0, "val": 0}


def test_the_validation_split_can_skip_the_rows_the_harness_evaluates_on(
    rukh_home: Path, repo_root: Path
) -> None:
    src = repo_root / "tests" / "fixtures" / "games.parquet"
    for month in ("01", "02"):
        target = rukh_home / "data" / "uci" / "year=2025" / f"month={month}" / "games.parquet"
        target.parent.mkdir(parents=True)
        shutil.copy(src, target)
    plain = build(PgnTextConfig(n_train=2, n_val=2)).counts["val"]
    out = rukh_home / "data" / "pgn-text"
    first = (out / "val.jsonl").read_text("utf-8")
    build(PgnTextConfig(n_train=2, n_val=2, val_offset=2))
    assert plain == 2
    assert (out / "val.jsonl").read_text("utf-8") != first


def test_missing_games_say_where_they_were_looked_for(rukh_home: Path) -> None:
    with pytest.raises(FileNotFoundError, match="games not found"):
        build(PgnTextConfig())
