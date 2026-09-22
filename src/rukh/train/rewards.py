"""Verifiable rewards for a chess move, and the reason each one is capped.

GRPO does not need a reward *model*: it needs a reward *function*, and a chess move is one of the
rare places where that function can be written down and checked. A move is legal or it is not; a
position is worth a number Stockfish will tell you; a mate is a mate. Nothing here is learned,
so nothing here can be wrong in the way a learned reward is wrong -- it can only be **designed**
wrong, which is a different problem and the one this module is about.

Four rules shape every function below.

**Legality is a gate, not a term.** An illegal move scores ``illegal_value(weights)`` -- a whole
unit below anything a legal move can reach, ``-1.25`` with the default weights -- and skips the
rest. Adding a legality *bonus* to a weighted sum lets the model trade legality for something
else, and the whole point of a verifiable reward is that some things are not tradeable. A flat
zero was the first answer and it was wrong; ``ILLEGAL_MARGIN`` below carries why (D-115).

**Everything is capped.** A reward with no ceiling is an invitation: the optimiser will find the
direction that grows without bound and go there, and what comes back is a model that maximises the
number instead of playing chess. Every function here returns a value in a closed interval, and
there is a test that says so over thousands of random inputs.

**Sign and scale are fixed by hand, not by the data.** `delta_cp` is normalised by a constant the
caller passes, not by the spread of the batch, because a reward that rescales itself makes two
runs incomparable and hides the moment the model stops improving.

**Every function is pure.** Board in, number out, no engine call inside. The engine scores are
passed in, which is what lets the tests run in milliseconds on the CPU and what keeps the reward
from becoming a second place where the analysis budget is decided.
"""

from __future__ import annotations

import chess
from pydantic import Field

from rukh.config import BaseConfig

__all__ = [
    "ILLEGAL_MARGIN",
    "RewardBreakdown",
    "RewardWeights",
    "illegal_value",
    "legal_bounds",
    "legality_gate",
    "mate_bonus",
    "normalised_delta_cp",
    "reward",
    "repetition_penalty",
]

ILLEGAL_MARGIN = 1.0
"""How far below every legal move an illegal one scores. See ``illegal_value``.

A flat ``0.0`` was the obvious choice and was wrong, and the gallery in `labs/m5` is what caught
it: with the default weights a legal move that walks into a repetition scores ``-0.25``, so a
zero for illegal moves put **illegal above legal** for exactly the moves the repetition term was
written to discourage. A gate that the thing it gates can score better than is not a gate.

The margin is a whole unit rather than an epsilon so the ordering survives any reweighting that
keeps the other terms in their documented ranges."""


class RewardWeights(BaseConfig):
    """How much each verifiable rule is worth, and the constants that cap them.

    The defaults are the ones the design spec 02 names, written down so a run can be reproduced from
    its config alone. They are **not** tuned: tuning them against the Elo they produce would make
    the reward a second model fitted to the evaluation, which is exactly the loop this module
    warns about.
    """

    quality: float = Field(default=1.0, ge=0)
    """Weight of the centipawn term: how close the move was to the best one available."""
    mate: float = Field(default=0.25, ge=0)
    """Weight of the mate bonus. Small on purpose -- finding mate is already the best `delta_cp`,
    so a large bonus would pay twice for the same move."""
    repetition: float = Field(default=0.25, ge=0)
    """Weight of the repetition penalty, the only negative term."""
    cp_scale: float = Field(default=200.0, gt=0)
    """Centipawns at which the quality term reaches zero. A move 200 cp worse than the best is as
    bad as this reward can say; beyond that the difference stops mattering, which is true of chess
    and keeps one catastrophe from dominating a whole group."""


class RewardBreakdown(BaseConfig):
    """The reward and every term that went into it, because a scalar hides its own design."""

    total: float
    legal: bool
    quality: float = 0.0
    mate: float = 0.0
    repetition: float = 0.0

    @property
    def terms(self) -> dict[str, float]:
        return {"quality": self.quality, "mate": self.mate, "repetition": self.repetition}


def legality_gate(board: chess.Board, move: chess.Move) -> bool:
    """Whether the move can be played here at all. The gate every other term sits behind."""
    return move in board.legal_moves


def normalised_delta_cp(cp_move: float, cp_best: float, scale: float = 200.0) -> float:
    """``1`` for the best move, ``0`` for one ``scale`` centipawns worse or more.

    Both scores are from the **mover's** point of view, so the best available move is the largest.
    Clamped at both ends: a move cannot be better than the best one, and one that is catastrophic
    is not more informative than one that is merely losing -- letting it go to minus infinity
    would let a single blunder decide the advantage of a whole group.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    loss = max(0.0, float(cp_best) - float(cp_move))
    return max(0.0, 1.0 - loss / scale)


def mate_bonus(board: chess.Board, move: chess.Move) -> float:
    """``1`` when the move mates, ``0`` otherwise. Deliberately binary.

    Not "distance to mate": a shorter mate is not a better outcome, it is the same outcome sooner,
    and paying for the difference teaches the model to prefer flashy mates over sure ones.
    """
    copy = board.copy(stack=False)
    copy.push(move)
    return 1.0 if copy.is_checkmate() else 0.0


def repetition_penalty(board: chess.Board, move: chess.Move) -> float:
    """``1`` when the move walks into a repetition, ``0`` otherwise.

    The one negative term, and the one that exists because of an observed failure rather than a
    principle: a model rewarded for not losing discovers that shuffling never loses. The penalty
    fires on a position seen before in this game, which is what `chess.Board.is_repetition(2)`
    answers, rather than on the threefold claim -- by the time the draw is claimable the shuffling
    has already happened and the reward arrives too late to discourage it.
    """
    copy = board.copy()
    copy.push(move)
    return 1.0 if copy.is_repetition(2) else 0.0


def reward(
    board: chess.Board,
    move: chess.Move,
    cp_move: float | None = None,
    cp_best: float | None = None,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    """The whole reward of one move, with its terms.

    ``cp_move`` and ``cp_best`` come from outside -- the caller decides the analysis budget once,
    for the whole group, instead of every reward call deciding it again. When they are missing the
    quality term is zero and the reward falls back to what can be known without an engine, which
    is legality, mate and repetition.
    """
    weights = weights or RewardWeights()
    if not legality_gate(board, move):
        return RewardBreakdown(total=illegal_value(weights), legal=False)

    quality = 0.0
    if cp_move is not None and cp_best is not None:
        quality = normalised_delta_cp(cp_move, cp_best, weights.cp_scale)
    mate = mate_bonus(board, move)
    repetition = repetition_penalty(board, move)

    total = weights.quality * quality + weights.mate * mate - weights.repetition * repetition
    return RewardBreakdown(
        total=total, legal=True, quality=quality, mate=mate, repetition=repetition
    )


def legal_bounds(weights: RewardWeights | None = None) -> tuple[float, float]:
    """The closed interval a **legal** move falls in."""
    weights = weights or RewardWeights()
    return (-weights.repetition, weights.quality + weights.mate)


def illegal_value(weights: RewardWeights | None = None) -> float:
    """What an illegal move scores: strictly below anything a legal move can reach.

    This is the number that makes legality a gate rather than a preference. It is derived from the
    weights instead of being a constant, because a constant is a promise that quietly stops being
    true the first time somebody raises ``repetition``.
    """
    return legal_bounds(weights)[0] - ILLEGAL_MARGIN


def bounds(weights: RewardWeights | None = None) -> tuple[float, float]:
    """The closed interval every reward falls in, which is what makes the design auditable."""
    return (illegal_value(weights), legal_bounds(weights)[1])
