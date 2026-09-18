"""Tests for rukh.eval.puzzles: whole-line scoring, partial lines, bands and loading."""

from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch

from rukh.eval.puzzles import (
    PuzzleAttempt,
    PuzzleItem,
    load_puzzles,
    model_source,
    run_puzzles,
    solve_puzzle,
    summarize,
)
from rukh.models import DecoderConfig, MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

pytestmark = pytest.mark.unit

TOY = DecoderConfig(vocab_size=2030, n_layer=2, n_head=2, d_model=32, block=64)
# The solver is Black: the opponent's moves are at even indices, the solution at odd ones.
TWO_MOVE_LINE = PuzzleItem(
    puzzle_id="two",
    fen=chess.STARTING_FEN,
    moves=["e2e4", "e7e5", "g1f3", "b8c6"],
    rating=1600,
    band="1500-2000",
)


@pytest.fixture(scope="module")
def tok() -> UciTokenizer:
    return UciTokenizer()


def scripted(moves: list[str | None]):  # type: ignore[no-untyped-def]
    """A move source that plays the given moves in order and then repeats the last one."""
    calls = {"n": 0}

    def choose(board: chess.Board, history: list[int]) -> chess.Move | None:
        move = moves[min(calls["n"], len(moves) - 1)]
        calls["n"] += 1
        return None if move is None else chess.Move.from_uci(move)

    return choose


def test_the_whole_line_must_be_played_to_solve_a_puzzle(tok: UciTokenizer) -> None:
    attempt = solve_puzzle(scripted(["e7e5", "b8c6"]), tok, TWO_MOVE_LINE)
    assert attempt.solved is True
    assert (attempt.correct, attempt.total) == (2, 2)


def test_a_partial_line_is_counted_but_not_solved(tok: UciTokenizer) -> None:
    attempt = solve_puzzle(scripted(["e7e5", "g8f6"]), tok, TWO_MOVE_LINE)
    assert attempt.solved is False
    assert (attempt.correct, attempt.total) == (1, 2)


def test_a_wrong_first_move_scores_nothing(tok: UciTokenizer) -> None:
    attempt = solve_puzzle(scripted(["a7a6"]), tok, TWO_MOVE_LINE)
    assert attempt.solved is False
    assert attempt.correct == 0


def test_a_source_that_declines_to_move_scores_nothing(tok: UciTokenizer) -> None:
    attempt = solve_puzzle(scripted([None]), tok, TWO_MOVE_LINE)
    assert attempt.solved is False
    assert attempt.correct == 0


def test_the_opponent_replies_are_forced_not_asked(tok: UciTokenizer) -> None:
    """Only the two solution moves are requested; the opponent's two moves are pushed."""
    asked: list[str] = []

    def choose(board: chess.Board, history: list[int]) -> chess.Move | None:
        asked.append(board.fen())
        return chess.Move.from_uci(["e7e5", "b8c6"][len(asked) - 1])

    solve_puzzle(choose, tok, TWO_MOVE_LINE)
    assert len(asked) == 2
    assert asked[0].split()[1] == "b"


def test_the_history_grows_with_every_move_of_the_line(tok: UciTokenizer) -> None:
    lengths: list[int] = []

    def choose(board: chess.Board, history: list[int]) -> chess.Move | None:
        lengths.append(len(history))
        return chess.Move.from_uci(["e7e5", "b8c6"][len(lengths) - 1])

    solve_puzzle(choose, tok, TWO_MOVE_LINE)
    assert lengths == [4, 6]  # header (3) + the opponent's move, then + 2 more


def test_summarize_reports_a_rate_per_band() -> None:
    attempts = [
        PuzzleAttempt(
            puzzle_id="a", band="1000-1500", rating=1200, solved=True, correct=1, total=1
        ),
        PuzzleAttempt(
            puzzle_id="b", band="1000-1500", rating=1300, solved=False, correct=0, total=1
        ),
        PuzzleAttempt(puzzle_id="c", band="2000+", rating=2100, solved=True, correct=2, total=2),
    ]
    result = summarize(attempts)
    assert result.attempted == 3
    assert result.solved == 2
    assert result.by_band() == {"1000-1500": 0.5, "2000+": 1.0}


def test_summarize_of_nothing_is_zero() -> None:
    assert summarize([]).rate == 0.0


def test_run_puzzles_folds_every_attempt(tok: UciTokenizer) -> None:
    other = TWO_MOVE_LINE.model_copy(update={"puzzle_id": "miss"})
    result = run_puzzles(scripted(["e7e5", "b8c6"]), tok, [TWO_MOVE_LINE, other])
    assert result.attempted == 2


def test_the_model_source_plays_a_legal_move(tok: UciTokenizer) -> None:
    torch.manual_seed(0)
    model = MoveDecoder(TOY).eval()
    board = chess.Board()
    move = model_source(model, tok)(board, [tok.bos_id])
    assert move is not None
    assert board.is_legal(move)


def test_load_puzzles_reads_the_test_split_band_by_band(tmp_path: Path) -> None:
    import polars as pl

    frame = pl.DataFrame(
        {
            "puzzle_id": ["a", "b", "c", "d"],
            "fen": [chess.STARTING_FEN] * 4,
            "moves": ["e2e4 e7e5"] * 4,
            "rating": [1200, 1300, 1700, 2100],
            "band": ["1000-1500", "1000-1500", "1500-2000", "2000+"],
            "split": ["test", "test", "test", "train"],
            "themes": [["mate"]] * 4,
        }
    )
    path = tmp_path / "puzzles.parquet"
    frame.write_parquet(path)
    items = load_puzzles(path, per_band=1, seed=0)
    assert sorted(item.band for item in items) == ["1000-1500", "1500-2000"]
    assert all(item.moves == ["e2e4", "e7e5"] for item in items)
