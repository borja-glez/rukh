"""Verifiable rewards for a chess move, and the reason each one is capped.

GRPO does not need a reward *model*: it needs a reward *function*, and a chess move is one of the
rare places where that function can be written down and checked. A move is legal or it is not; a
position is worth a number Stockfish will tell you; a mate is a mate. Nothing here is learned,
so nothing here can be wrong in the way a learned reward is wrong -- it can only be **designed**
wrong, which is a different problem and the one this module is about.

Four rules shape every function below.

**Legality is a gate, not a term.** An illegal move scores zero and skips the rest. Adding a
legality *bonus* to a weighted sum lets the model trade legality for something else, and the whole
point of a verifiable reward is that some things are not tradeable.

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
    "ILLEGAL",
    "RewardBreakdown",
    "RewardWeights",
    "legality_gate",
    "mate_bonus",
    "normalised_delta_cp",
    "reward",
    "repetition_penalty",
]

ILLEGAL = 0.0
"""What an illegal move scores, before anything else is computed."""


class RewardWeights(BaseConfig):
    """How much each verifiable rule is worth, and the constants that cap them.

    The defaults are the ones `docs/spec/02` names, written down so a run can be reproduced from
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
        return RewardBreakdown(total=ILLEGAL, legal=False)

    quality = 0.0
    if cp_move is not None and cp_best is not None:
        quality = normalised_delta_cp(cp_move, cp_best, weights.cp_scale)
    mate = mate_bonus(board, move)
    repetition = repetition_penalty(board, move)

    total = weights.quality * quality + weights.mate * mate - weights.repetition * repetition
    return RewardBreakdown(
        total=total, legal=True, quality=quality, mate=mate, repetition=repetition
    )


def bounds(weights: RewardWeights | None = None) -> tuple[float, float]:
    """The closed interval every reward falls in, which is what makes the design auditable."""
    weights = weights or RewardWeights()
    return (-weights.repetition, weights.quality + weights.mate)
