"""The dumb baseline the encoder has to beat: material, mobility and a one-ply look at captures.

``docs/acceptance.md`` asks for a blunder F1 five points above a heuristic, and the heuristic is
only worth
anything as a comparison if it is measured exactly like the model: same positions, same labels,
same definition of a hit. So it lives here, next to the suite, and is deliberately kept as
simple as a chess program can be:

* **material**: pawn 1, knight 3, bishop 3, rook 5, queen 9, king 0 — the values every beginner
  is taught, unchanged by the phase of the game;
* **mobility**: how many legal moves each side has, worth ``MOBILITY_WEIGHT`` of a pawn each,
  which is what turns a pile of material into something that moves with the position;
* **blunder**: the move that led here lost at least ``BLUNDER_MATERIAL`` point of net material,
  measured from the mover's point of view, by playing the move on a ``python-chess`` board and
  letting the opponent answer with its single most profitable capture or promotion.

What it cannot see
------------------

Everything that is not material. The named example the tests use is **Byrne–Fischer, New York
1956** (the "Game of the Century"), after 17.Kf1: Fischer played 17...Be6, offering the queen,
and the game is a win for Black. The heuristic sees 18.Bxb6 taking a queen for nothing and calls
one of the most famous moves in chess a blunder. A positional sacrifice is invisible to it, and
that is the point: a baseline that already understood sacrifices would not be a baseline.

It is also blind in the other direction. The opponent's reply is searched one ply deep and only
over captures and promotions, so our own recapture is never counted: an equal trade whose
recapture comes two plies later reads as a loss. And a piece that was already hanging before the
move is charged again to every quiet move that follows it, because "before" is the material on
the board and not the best the mover could have kept. Both are written down here rather than
fixed, because fixing them would make the baseline a small engine instead of a floor.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import chess
from pydantic import BaseModel, ConfigDict

PIECE_VALUES: dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
"""The beginner's table, in pawns. The king is worth nothing: it is never captured."""

MOBILITY_WEIGHT = 0.05
"""Pawns per legal move of difference; twenty extra moves are worth one pawn."""

BLUNDER_MATERIAL = 1.0
"""Net material a move has to lose, in pawns, before the baseline calls it a blunder."""

VALUE_SCALE = 4.0
"""``tanh(score / 4)`` in pawns is exactly ``tanh(cp / 400)``, the label's own scale."""

GAME_OF_THE_CENTURY = "r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3R1K1R b - - 3 17"
"""Byrne–Fischer, New York 1956, after 17.Kf1: the position of the positional sacrifice."""

POSITIONAL_SACRIFICE = "g4e6"
"""17...Be6, the queen offer the heuristic reads as a nine-point blunder."""


class Verdict(BaseModel):
    """What the baseline says about one move: a value, a material loss and a verdict."""

    model_config = ConfigDict(extra="forbid")

    move: str
    blunder: bool
    loss: float
    """Net material the mover lost, in pawns; negative when the move won material."""
    material_before: float
    material_after: float
    """Material after the move and the opponent's most profitable capture."""
    value: float
    """Value of the resulting position, ``tanh(score / 4)`` from White's point of view."""


def material(board: chess.Board) -> float:
    """Material on the board in pawns, from White's point of view."""
    total = 0.0
    for piece_type, value in PIECE_VALUES.items():
        if not value:
            continue
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total -= value * len(board.pieces(piece_type, chess.BLACK))
    return total


def material_for(board: chess.Board, color: chess.Color) -> float:
    """Material from ``color``'s point of view: its own minus the opponent's."""
    return material(board) if color == chess.WHITE else -material(board)


def _moves_for(board: chess.Board, color: chess.Color) -> int:
    """How many legal moves ``color`` has, whoever is to move."""
    if board.turn == color:
        return board.legal_moves.count()
    swapped = board.copy(stack=False)
    swapped.turn = color
    swapped.ep_square = None
    return swapped.legal_moves.count()


def mobility(board: chess.Board) -> int:
    """White's legal moves minus Black's, counted on the same position."""
    return _moves_for(board, chess.WHITE) - _moves_for(board, chess.BLACK)


def score(board: chess.Board, mobility_weight: float = MOBILITY_WEIGHT) -> float:
    """Material plus weighted mobility, in pawns, from White's point of view."""
    return material(board) + mobility_weight * mobility(board)


def value(board: chess.Board, mobility_weight: float = MOBILITY_WEIGHT) -> float:
    """The baseline's value of a position, on the label's own ``tanh`` scale."""
    return math.tanh(score(board, mobility_weight) / VALUE_SCALE)


def value_of(fen: str, mobility_weight: float = MOBILITY_WEIGHT) -> float:
    """``value`` of a FEN (four fields or six)."""
    return value(chess.Board(fen), mobility_weight)


def _material_moves(board: chess.Board) -> Iterator[chess.Move]:
    """The only replies that can change the material count: captures and promotions."""
    yield from board.generate_legal_captures()
    for move in board.legal_moves:
        if move.promotion is not None and not board.is_capture(move):
            yield move


def worst_material(board: chess.Board, color: chess.Color) -> float:
    """``color``'s material after the single most profitable reply the opponent has.

    One ply, captures and promotions only: no recapture, no threat, no check. That is the whole
    search, and the docstring of the module says what it costs.
    """
    worst = material_for(board, color)
    for reply in _material_moves(board):
        child = board.copy(stack=False)
        child.push(reply)
        worst = min(worst, material_for(child, color))
    return worst


def judge(
    fen: str,
    move: str,
    threshold: float = BLUNDER_MATERIAL,
    mobility_weight: float = MOBILITY_WEIGHT,
) -> Verdict:
    """Judge ``move`` played in ``fen``: how much material it lost and whether that is a blunder.

    ``fen`` is the position **before** the move, which is the only way material can be compared;
    the labels of ``rukh.data.labels`` judge the same move from the same side.
    """
    board = chess.Board(fen)
    mover = board.turn
    before = material_for(board, mover)
    board.push(chess.Move.from_uci(move))
    after = worst_material(board, mover)
    loss = before - after
    return Verdict(
        move=move,
        blunder=loss >= threshold,
        loss=loss,
        material_before=before,
        material_after=after,
        value=value(board, mobility_weight),
    )
