"""Tests for rukh.eval.heuristic: what the material baseline sees and what it cannot see."""

from __future__ import annotations

import math

import chess
import pytest

from rukh.eval.heuristic import (
    GAME_OF_THE_CENTURY,
    PIECE_VALUES,
    POSITIONAL_SACRIFICE,
    judge,
    material,
    material_for,
    mobility,
    score,
    value,
    value_of,
)

pytestmark = pytest.mark.unit

HANGING_QUEEN = "r1bqkbnr/pppp1ppp/2n5/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 2 3"
"""After 1.e4 e5 2.Qh5 Nc6: 3.Qxf7+ wins a pawn and loses the queen to 3...Kxf7."""


def test_the_starting_position_is_perfectly_balanced() -> None:
    board = chess.Board()
    assert material(board) == 0.0
    assert mobility(board) == 0  # twenty moves each
    assert value(board) == 0.0


def test_material_is_counted_from_the_asking_side() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert material(board) == PIECE_VALUES[chess.QUEEN]
    assert material_for(board, chess.WHITE) == 9.0
    assert material_for(board, chess.BLACK) == -9.0


def test_mobility_moves_the_score_without_moving_the_material() -> None:
    # A rook on an open file has more moves than one boxed in, with the same pieces on the board.
    open_file = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    boxed_in = chess.Board("4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
    assert material(open_file) == material(boxed_in)
    assert score(open_file) != score(boxed_in)


def test_the_baseline_flags_a_hanging_queen() -> None:
    verdict = judge(HANGING_QUEEN, "h5f7")
    assert verdict.blunder is True
    assert verdict.loss == pytest.approx(8.0)  # a queen for a pawn
    assert verdict.material_before == pytest.approx(0.0)


def test_a_quiet_developing_move_is_not_a_blunder() -> None:
    assert judge(HANGING_QUEEN, "g1f3").blunder is False
    assert judge(HANGING_QUEEN, "g1f3").loss == pytest.approx(0.0)


def test_an_even_trade_is_not_a_blunder() -> None:
    # Ruy Lopez, 4.Bxc6: a bishop for a knight, recaptured at once. The one ply of replies is
    # exactly what keeps an even trade off the blunder list.
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6"):
        board.push_san(san)
    verdict = judge(board.fen(), "b5c6")
    assert verdict.blunder is False
    assert verdict.loss == pytest.approx(0.0)


def test_losing_a_piece_for_a_pawn_is_a_blunder() -> None:
    # 1.e4 e5 2.Nf3 Nc6 3.Nxe5?? Nxe5: the knight wins a pawn and is lost for nothing.
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6"):
        board.push_san(san)
    verdict = judge(board.fen(), "f3e5")
    assert verdict.blunder is True
    assert verdict.loss == pytest.approx(2.0)  # a knight for a pawn


def test_the_baseline_cannot_see_a_positional_sacrifice() -> None:
    """Byrne-Fischer 1956, 17...Be6: the Game of the Century, scored as a nine-point blunder.

    This is the documented blind spot of the baseline and the reason the encoder is expected to
    beat it: material plus one ply of captures cannot tell a queen sacrifice from a queen loss.
    """
    verdict = judge(GAME_OF_THE_CENTURY, POSITIONAL_SACRIFICE)
    assert verdict.blunder is True
    assert verdict.loss == pytest.approx(9.0)
    # Black is a pawn up before the move and eight points down after 18.Bxb6, so the baseline
    # cannot help calling it: the compensation is not on the board yet. Fischer won the game.
    assert verdict.material_before == pytest.approx(1.0)
    assert verdict.material_after == pytest.approx(-8.0)


def test_the_value_is_on_the_label_s_own_scale() -> None:
    # The labels are tanh(cp / 400) and the baseline is tanh(pawns / 4), so four pawns of
    # advantage (400 centipawns) read as tanh(1) on both sides of the comparison.
    four_pawns_up = chess.Board("4k3/8/8/8/8/8/P7/1N2K3 w - - 0 1")
    assert value(four_pawns_up, mobility_weight=0.0) == pytest.approx(math.tanh(1.0), abs=1e-12)
    assert value_of(four_pawns_up.fen(), 0.0) == value(four_pawns_up, 0.0)


def test_a_promotion_is_material_even_when_nothing_is_captured() -> None:
    # Black to move cannot stop the pawn: the mover's quiet king move hands White a queen.
    board = chess.Board("8/P7/8/8/8/8/8/k5K1 b - - 0 1")
    verdict = judge(board.fen(), "a1b1")
    assert verdict.blunder is True
    assert verdict.loss == pytest.approx(8.0)  # a queen appears, minus the pawn that became it
