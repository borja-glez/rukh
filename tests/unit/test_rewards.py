"""The reward functions, written before anything trains on them.

A learned reward can be wrong in ways nobody notices. A verifiable one can only be **designed**
wrong, and a design is testable: the best move gets the most, an illegal move gets nothing, and no
input anywhere produces a number outside the interval the design promises.
"""

from __future__ import annotations

import random

import chess
import pytest

from rukh.train.rewards import (
    ILLEGAL,
    RewardWeights,
    bounds,
    legality_gate,
    mate_bonus,
    normalised_delta_cp,
    repetition_penalty,
    reward,
)

pytestmark = pytest.mark.unit


def test_legality_is_a_gate_and_not_a_term():
    """An illegal move scores zero flat, with no other term able to buy it back."""
    board = chess.Board()
    illegal = chess.Move.from_uci("e2e5")
    assert legality_gate(board, illegal) is False

    got = reward(board, illegal, cp_move=1000, cp_best=0)
    assert got.total == ILLEGAL
    assert got.legal is False
    # Even with the best possible centipawn score behind it, nothing leaks through the gate.
    assert got.terms == {"quality": 0.0, "mate": 0.0, "repetition": 0.0}


def test_the_best_move_scores_one_and_the_scale_is_where_it_runs_out():
    assert normalised_delta_cp(50, 50) == 1.0
    assert normalised_delta_cp(-150, 50, scale=200) == 0.0
    assert normalised_delta_cp(-50, 50, scale=200) == pytest.approx(0.5)
    # Better than the best is not a thing; the clamp says so instead of producing 1.2.
    assert normalised_delta_cp(100, 50, scale=200) == 1.0
    # And a catastrophe is not more informative than a loss: both bottom out.
    assert normalised_delta_cp(-100_000, 50, scale=200) == 0.0


def test_a_scale_of_zero_is_refused_rather_than_dividing():
    with pytest.raises(ValueError, match="scale must be positive"):
        normalised_delta_cp(0, 100, scale=0)


def test_mate_is_binary_and_does_not_pay_for_being_quick():
    """Fool's mate: `Qh5xf7#` ends it, and a shorter mate is the same outcome sooner."""
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4"):
        board.push_uci(uci)
    mate = chess.Move.from_uci("d8h4")
    assert mate_bonus(board, mate) == 1.0
    assert mate_bonus(board, chess.Move.from_uci("g8f6")) == 0.0


def test_the_repetition_penalty_fires_on_the_second_visit_not_on_the_claim():
    """By the time a draw is claimable the shuffling already happened; the penalty is too late."""
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    # The starting position is on the board again, so `Nf3` returns to a position already seen.
    assert repetition_penalty(board, chess.Move.from_uci("g1f3")) == 1.0
    assert repetition_penalty(board, chess.Move.from_uci("e2e4")) == 0.0


def test_the_reward_carries_its_terms_because_a_scalar_hides_its_design():
    board = chess.Board()
    got = reward(board, chess.Move.from_uci("e2e4"), cp_move=20, cp_best=30)
    assert got.legal is True
    assert got.quality == pytest.approx(1.0 - 10 / 200)
    assert got.total == pytest.approx(got.quality)
    assert set(got.terms) == {"quality", "mate", "repetition"}


def test_without_engine_scores_the_reward_falls_back_to_what_is_knowable():
    """No `cp` means no quality term -- not a guessed one."""
    board = chess.Board()
    got = reward(board, chess.Move.from_uci("e2e4"))
    assert got.legal is True
    assert got.quality == 0.0
    assert got.total == 0.0


def test_no_input_escapes_the_interval_the_design_promises():
    """A reward with no ceiling is an invitation; this is the test that keeps the promise.

    Random legal moves from random positions, with centipawn scores drawn far outside anything an
    engine would return, including the mate scores the pipeline uses.
    """
    weights = RewardWeights()
    low, high = bounds(weights)
    rng = random.Random(7)
    seen = 0
    for _ in range(300):
        board = chess.Board()
        for _ in range(rng.randrange(0, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        moves = list(board.legal_moves)
        if not moves:
            continue
        move = rng.choice(moves)
        cp_move = rng.choice([-10_000, -999, 0, 37, 10_000])
        cp_best = rng.choice([-10_000, -12, 0, 500, 10_000])
        total = reward(board, move, cp_move, cp_best, weights).total
        assert low <= total <= high, (board.fen(), move.uci(), cp_move, cp_best, total)
        seen += 1
    assert seen > 200, "the walk should have produced plenty of legal positions"


def test_the_bounds_move_with_the_weights_instead_of_being_a_constant():
    weights = RewardWeights(quality=2.0, mate=0.5, repetition=1.0)
    assert bounds(weights) == (-1.0, 2.5)


def test_the_weights_are_refused_when_they_would_break_the_design():
    with pytest.raises(ValueError):
        RewardWeights(quality=-1.0)
    with pytest.raises(ValueError):
        RewardWeights(cp_scale=0)
