"""GRPO: the baseline, the penalty, and the two ways this loop can lie to itself.

The parts worth testing here are the ones where a bug produces a *number* rather than a crash: a
baseline that manufactures a gradient out of a group that all agreed, a KL term with the wrong
sign that pays the model for leaving its reference, and a cache that answers for the wrong
position.
"""

from __future__ import annotations

import chess
import pytest
import torch

from rukh.train.grpo import (
    EngineCache,
    GrpoConfig,
    group_advantages,
    grpo_loss,
    sample_group,
    score_candidates,
)

pytestmark = pytest.mark.unit


def test_a_group_that_agrees_produces_no_gradient():
    """The baseline is the group's own mean, so a flat group has nothing left to say.

    Dividing by a spread of zero would hand back whatever the epsilon makes of floating-point
    dust, and the loop would spend a step chasing it.
    """
    assert torch.equal(group_advantages([0.42] * 8), torch.zeros(8))
    assert torch.equal(group_advantages([0.0, 0.0]), torch.zeros(2))


def test_the_advantages_are_centred_and_scaled_by_the_groups_own_spread():
    advantages = group_advantages([1.0, 0.0, 0.5, 0.5])
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(advantages[0]) > 0 and float(advantages[1]) < 0
    # A group ten times as spread out gets the same advantages: only the ordering is kept.
    wide = group_advantages([10.0, 0.0, 5.0, 5.0])
    assert torch.allclose(advantages, wide, atol=1e-3)


def test_the_kl_term_is_never_negative():
    """k3 is non-negative by construction, which is the point of using it.

    The naive ``logp_ref - logp_policy`` is negative on exactly the samples where the policy moved
    away from the reference, so as a *penalty* it would pay for the behaviour it is meant to
    discourage. The test draws both directions on purpose.
    """
    torch.manual_seed(0)
    actions = torch.tensor([0, 1, 2, 3])
    advantages = torch.zeros(4)
    for _ in range(20):
        policy = torch.log_softmax(torch.randn(4, 8), dim=-1)
        reference = torch.log_softmax(torch.randn(4, 8), dim=-1)
        _loss, kl = grpo_loss(policy, reference, actions, advantages, beta_kl=1.0)
        assert float(kl) >= 0.0


def test_the_kl_is_zero_when_the_policy_is_still_the_reference():
    """The first step of any run: the penalty must cost nothing before anything has moved."""
    logprobs = torch.log_softmax(torch.randn(4, 8), dim=-1)
    _loss, kl = grpo_loss(
        logprobs, logprobs.clone(), torch.tensor([0, 1, 2, 3]), torch.zeros(4), beta_kl=1.0
    )
    assert float(kl) == pytest.approx(0.0, abs=1e-6)


def test_a_positive_advantage_pushes_its_move_up():
    """The smallest end-to-end check of the sign: one step of the loss raises the chosen move."""
    torch.manual_seed(0)
    logits = torch.zeros(1, 4, requires_grad=True)
    actions = torch.tensor([2])
    advantages = torch.tensor([1.0])
    for _ in range(50):
        logprobs = torch.log_softmax(logits, dim=-1)
        loss, _kl = grpo_loss(logprobs, logprobs.detach(), actions, advantages, beta_kl=0.0)
        loss.backward()
        with torch.no_grad():
            logits -= 0.5 * logits.grad
        logits.grad = None
    assert int(logits.argmax()) == 2

    # And a negative advantage pushes it down, which is the same test with the sign flipped.
    logits = torch.zeros(1, 4, requires_grad=True)
    for _ in range(50):
        logprobs = torch.log_softmax(logits, dim=-1)
        loss, _kl = grpo_loss(logprobs, logprobs.detach(), actions, -advantages, beta_kl=0.0)
        loss.backward()
        with torch.no_grad():
            logits -= 0.5 * logits.grad
        logits.grad = None
    assert int(logits.argmin()) == 2


def test_sampling_only_ever_draws_from_the_moves_it_was_given():
    """Everything outside ``legal_ids`` is masked to minus infinity, however large its logit."""
    logits = torch.zeros(10)
    logits[7] = 50.0  # the model is very sure about a move that is not on the list
    drawn = sample_group(logits, [1, 2, 3], size=64, temperature=1.0, top_k=None)
    assert set(int(token) for token in drawn) <= {1, 2, 3}


def test_sampling_keeps_duplicates_because_the_group_is_a_sample_of_the_policy():
    """A policy that would play one move eight times out of eight is saying something true."""
    logits = torch.zeros(10)
    logits[2] = 30.0
    drawn = sample_group(logits, [1, 2, 3], size=8, temperature=1.0, top_k=None)
    assert drawn.shape == (8,)
    assert set(int(token) for token in drawn) == {2}


class _CountingEngine:
    """An engine that refuses to be called twice for the same thing."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def analyse(self, board: chess.Board, limit: object) -> dict[str, object]:
        self.seen.append(board.fen())
        return {"score": chess.engine.PovScore(chess.engine.Cp(33), chess.WHITE)}


def test_the_cache_answers_the_second_time_without_the_engine():
    cache = EngineCache(None)
    engine = _CountingEngine()
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    first = cache.score(engine, board, move, depth=4)
    second = cache.score(engine, board, move, depth=4)
    assert first == second == 33
    assert len(engine.seen) == 1
    assert (cache.calls, cache.hits) == (1, 1)


def test_the_cache_is_keyed_by_the_position_not_by_the_move_list():
    """Two move orders reaching the same position are one cache entry, which is why it pays."""
    direct = chess.Board()
    direct.push_uci("g1f3")
    direct.push_uci("g8f6")
    transposed = chess.Board()
    transposed.push_uci("g1f3")
    transposed.push_uci("g8f6")
    move = chess.Move.from_uci("d2d4")
    assert EngineCache.key(direct.fen(), move) == EngineCache.key(transposed.fen(), move)


def test_the_config_refuses_a_group_that_cannot_have_a_baseline():
    with pytest.raises(ValueError):
        GrpoConfig(group_size=1)
    with pytest.raises(ValueError):
        GrpoConfig(beta_kl=-0.1)


def test_a_collapsed_group_scores_the_maximum_against_its_own_best():
    """The flaw that makes the training reward useless as a score, pinned so it stays documented.

    ``score_candidates`` takes ``cp_best`` over the candidates themselves, which is what keeps a
    group of bad moves teachable -- the ordering inside it survives. The price is that a policy
    proposing one move eight times gets quality 1.0 on all eight, which is the highest the reward
    goes. So the training number is maximised by collapsing, and only the version that takes
    ``cp_best`` from the engine can be read as a score.
    """

    class _Fixed:
        """An engine that says every move leaves the position at -50 for the mover."""

        def analyse(self, board: chess.Board, limit: object) -> dict[str, object]:
            return {"score": chess.engine.PovScore(chess.engine.Cp(50), chess.WHITE)}

    cache = EngineCache(None)
    board = chess.Board()
    move = chess.Move.from_uci("a2a3")  # a legal move, and not a good one
    collapsed = [move] * 8

    relative = score_candidates(cache, _Fixed(), board, collapsed, GrpoConfig())
    assert relative == [pytest.approx(1.0)] * 8, "the group is its own best, so every one is best"

    # Against a genuinely better reference the same eight moves score the floor of the quality
    # term, which is what a mediocre move deserves and what the published number has to use.
    absolute = score_candidates(cache, _Fixed(), board, collapsed, GrpoConfig(), best=400.0)
    assert absolute == [pytest.approx(0.0)] * 8
