"""Tests for rukh.eval.diversity: the opening-entropy metric the spec asks for.

It is the second half of the claim a masters fine-tune makes -- "the Elo goes up and the
diversity goes down" -- so it has to be checkable on its own, with distributions whose entropy is
known by hand.
"""

from __future__ import annotations

import math
from collections import Counter

import chess
import pytest
import torch

from rukh.eval.diversity import (
    DiversityResult,
    first_move_entropy,
    opening_diversity,
    shannon_entropy,
)
from rukh.eval.legality import header
from rukh.infer.sampler import SampleConfig, legal_token_ids
from rukh.models import DecoderConfig, MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

pytestmark = pytest.mark.unit


def _model() -> MoveDecoder:
    torch.manual_seed(0)
    return MoveDecoder(DecoderConfig(n_layer=2, n_head=2, d_model=32, block=64)).eval()


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"a": 8}, 0.0),  # one opening, always
        ({"a": 1, "b": 1}, 1.0),  # a fair coin is one bit
        ({"a": 1, "b": 1, "c": 1, "d": 1}, 2.0),  # four equal outcomes are two
        ({"a": 3, "b": 1}, 0.8112781244591328),  # the textbook 3:1 value
        ({}, 0.0),
    ],
    ids=["single", "coin", "four", "skewed", "empty"],
)
def test_entropy_matches_the_values_computed_by_hand(
    counts: dict[str, int], expected: float
) -> None:
    assert shannon_entropy(Counter(counts)) == pytest.approx(expected)


def test_first_move_entropy_is_below_the_twenty_move_ceiling() -> None:
    """A model with no opinion would sit at log2(20); a real one has to be under it."""
    tok = UciTokenizer()
    entropy = first_move_entropy(_model(), tok)
    assert 0.0 <= entropy <= math.log2(20) + 1e-9


def test_first_move_entropy_is_zero_when_the_model_is_certain() -> None:
    """The collapse a narrow repertoire produces, forced by hand on the logits."""
    tok = UciTokenizer()
    model = _model()
    chosen = tok.vocab["e2e4"]

    class Certain(MoveDecoder):
        def next_logits(self, idx: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = torch.full((idx.shape[0], model.cfg.vocab_size), -50.0)
            out[:, chosen] = 50.0
            return out

    certain = Certain(model.cfg).eval()
    assert first_move_entropy(certain, tok) == pytest.approx(0.0, abs=1e-6)


def test_first_move_entropy_only_counts_legal_moves() -> None:
    """Twenty moves are legal at the start; the other 2 010 tokens must not dilute the measure."""
    tok = UciTokenizer()
    assert len(legal_token_ids(chess.Board(), tok)) == 20
    model = _model()
    with torch.no_grad():
        flat = torch.zeros(model.cfg.vocab_size)
    assert shannon_entropy(Counter({str(i): 1 for i in range(20)})) == pytest.approx(
        first_move_entropy(_Flat(model.cfg, flat).eval(), tok), abs=1e-5
    )


class _Flat(MoveDecoder):
    """A decoder whose next-token distribution is uniform over the whole vocabulary."""

    def __init__(self, cfg: DecoderConfig, logits: torch.Tensor) -> None:
        super().__init__(cfg)
        self._flat = logits

    def next_logits(self, idx: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self._flat.expand(idx.shape[0], -1)


def test_deterministic_sampling_gives_one_line_and_zero_entropy() -> None:
    """True, and it is why the metric always travels with the temperature it was read at.

    At the setting the demo plays at (T=0.05, top-k 1) a decoder plays the same game every time,
    so the number says more about the sampler than about the weights (D-047).
    """
    tok = UciTokenizer()
    result = opening_diversity(
        _model(), tok, games=8, plies=4, sampling=SampleConfig(temperature=0.05, top_k=1)
    )
    assert result.distinct_lines == 1
    assert result.entropy_bits == pytest.approx(0.0)
    assert result.normalised == pytest.approx(0.0)
    assert result.temperature == 0.05


def test_sampling_opens_the_fan_out() -> None:
    tok = UciTokenizer()
    result = opening_diversity(
        _model(), tok, games=16, plies=4, sampling=SampleConfig(temperature=1.0, top_k=20)
    )
    assert result.distinct_lines > 1
    assert 0.0 < result.entropy_bits <= result.max_entropy_bits
    assert result.max_entropy_bits == pytest.approx(4.0)  # log2(16)


def test_every_line_is_a_sequence_of_legal_moves() -> None:
    """The metric measures which openings are chosen, so the mask is always on."""
    tok = UciTokenizer()
    result = opening_diversity(
        _model(), tok, games=6, plies=6, sampling=SampleConfig(temperature=1.0, top_k=20)
    )
    for line, _ in result.top_lines:
        board = chess.Board()
        for uci in line.split():
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            board.push(move)


def test_the_result_is_reproducible_for_a_seed() -> None:
    tok = UciTokenizer()
    model = _model()
    sampling = SampleConfig(temperature=1.0, top_k=20)
    first = opening_diversity(model, tok, games=8, plies=4, sampling=sampling, seed=7)
    second = opening_diversity(model, tok, games=8, plies=4, sampling=sampling, seed=7)
    assert first == second
    other = opening_diversity(model, tok, games=8, plies=4, sampling=sampling, seed=8)
    assert other.top_lines != first.top_lines or other.entropy_bits != first.entropy_bits


def test_the_header_the_games_start_from_is_the_one_asked_for() -> None:
    tok = UciTokenizer()
    assert header(tok, 1500, 1500)[1:] == [tok.vocab["<w1500>"], tok.vocab["<b1500>"]]
    result = opening_diversity(
        _model(), tok, games=4, plies=2, sampling=SampleConfig(temperature=1.0), header_elo=1500
    )
    assert isinstance(result, DiversityResult)
    assert result.games == 4


def test_the_first_move_distribution_is_over_the_twenty_legal_moves_and_sums_to_one() -> None:
    """The instrument that measures what a style adapter did, with no seed in it."""
    from rukh.eval.diversity import first_move_distribution

    tok = UciTokenizer()
    probs = first_move_distribution(_model(), tok)
    assert len(probs) == 20
    assert set(probs) >= {"e2e4", "d2d4", "g1f3", "b1c3"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_the_distribution_and_the_entropy_agree() -> None:
    from rukh.eval.diversity import first_move_distribution

    tok = UciTokenizer()
    model = _model()
    probs = first_move_distribution(model, tok)
    by_hand = -sum(p * math.log2(p) for p in probs.values() if p > 0)
    assert first_move_entropy(model, tok) == pytest.approx(by_hand, abs=1e-6)


def test_a_collapsed_repertoire_shows_up_as_a_share_near_one() -> None:
    """What a style adapter is supposed to do, forced by hand so the metric can be checked."""
    from rukh.eval.diversity import first_move_distribution

    tok = UciTokenizer()
    base = _model()
    chosen = tok.vocab["d2d4"]

    class Narrow(MoveDecoder):
        def next_logits(self, idx: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = torch.full((idx.shape[0], base.cfg.vocab_size), -20.0)
            out[:, chosen] = 20.0
            return out

    probs = first_move_distribution(Narrow(base.cfg).eval(), tok)
    assert probs["d2d4"] == pytest.approx(1.0, abs=1e-6)
    assert probs["e2e4"] < 1e-6


def test_a_single_outcome_is_zero_and_not_negative_zero() -> None:
    """`-1 * log2(1)` is negative zero, and a report that prints "-0.000" looks like a bug."""
    from rukh.eval.diversity import shannon_entropy

    value = shannon_entropy(Counter({"a": 200}))
    assert value == 0.0
    assert f"{value:.3f}" == "0.000"
