"""Tests for rukh.eval.accuracy: Elo bands and top-1/top-3 agreement with the human move."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from rukh.eval.accuracy import accuracy, elo_band
from rukh.eval.legality import position_at
from rukh.models import DecoderConfig, MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

pytestmark = pytest.mark.unit

TOY = DecoderConfig(vocab_size=2030, n_layer=2, n_head=2, d_model=32, block=64)
GAME = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6"


@pytest.fixture(scope="module")
def tok() -> UciTokenizer:
    return UciTokenizer()


class RankedModel:
    """A stand-in for the decoder whose logits rank a fixed list of tokens first."""

    def __init__(self, tok: UciTokenizer, ranking: list[str]) -> None:
        self.ids = [tok.vocab[token] for token in ranking]
        self.vocab_size = len(tok)
        self.cfg = TOY  # the prompt is cropped to the model's block, so the stub needs one

    def parameters(self) -> Iterator[torch.Tensor]:
        return iter([torch.zeros(1)])

    def next_logits(self, idx: torch.Tensor) -> torch.Tensor:
        logits = torch.full((idx.shape[0], self.vocab_size), -10.0)
        for rank, token_id in enumerate(self.ids):
            logits[:, token_id] = 10.0 - rank
        return logits


@pytest.mark.parametrize(
    ("elo", "expected"),
    [
        (1200, "<1800"),
        (1799, "<1800"),
        (1800, "1800-2000"),
        (1999, "1800-2000"),
        (2000, "2000-2200"),
        (2599, "2400-2600"),
        (2600, "2600+"),
        (3000, "2600+"),
    ],
)
def test_elo_band_cuts_every_two_hundred_points(elo: int, expected: str) -> None:
    assert elo_band(elo) == expected


def test_top1_and_top3_see_the_ranking(tok: UciTokenizer) -> None:
    moves = GAME.split()
    positions = [position_at(tok, 1, moves, 1, 1850, 1950)]  # the human played e7e5
    exact = accuracy(RankedModel(tok, ["e7e5", "a7a6", "b8c6"]), tok, positions)  # type: ignore[arg-type]
    assert (exact.top1, exact.top3) == (1.0, 1.0)
    third = accuracy(RankedModel(tok, ["a7a6", "b8c6", "e7e5"]), tok, positions)  # type: ignore[arg-type]
    assert (third.top1, third.top3) == (0.0, 1.0)
    missed = accuracy(RankedModel(tok, ["a7a6", "b8c6", "g8f6"]), tok, positions)  # type: ignore[arg-type]
    assert (missed.top1, missed.top3) == (0.0, 0.0)


def test_positions_are_filed_under_the_band_of_the_side_to_move(tok: UciTokenizer) -> None:
    moves = GAME.split()
    positions = [
        position_at(tok, 1, moves, 1, 1850, 2150),  # Black to move: 2150 -> 2000-2200
        position_at(tok, 1, moves, 2, 1850, 2150),  # White to move: 1850 -> 1800-2000
    ]
    result = accuracy(RankedModel(tok, ["e7e5"]), tok, positions)  # type: ignore[arg-type]
    bands = {band.band: band for band in result.bands}
    assert set(bands) == {"1800-2000", "2000-2200"}
    assert bands["2000-2200"].top1 == 1.0
    assert bands["1800-2000"].top1 == 0.0
    assert result.top1 == pytest.approx(0.5)


def test_positions_without_a_human_move_are_skipped(tok: UciTokenizer) -> None:
    moves = GAME.split()
    end = position_at(tok, 1, moves, len(moves), 1850, 1950)
    assert end.target is None
    assert accuracy(RankedModel(tok, ["e7e5"]), tok, [end]).positions == 0  # type: ignore[arg-type]


def test_accuracy_of_nothing_is_zero(tok: UciTokenizer) -> None:
    result = accuracy(RankedModel(tok, ["e7e5"]), tok, [])  # type: ignore[arg-type]
    assert (result.positions, result.top1, result.top3) == (0, 0.0, 0.0)


def test_accuracy_runs_on_the_real_decoder(tok: UciTokenizer) -> None:
    torch.manual_seed(0)
    model = MoveDecoder(TOY).eval()
    moves = GAME.split()
    positions = [position_at(tok, 1, moves, ply, 1850, 1950) for ply in range(1, 6)]
    result = accuracy(model, tok, positions)
    assert result.positions == 5
    assert result.top1 <= result.top3
