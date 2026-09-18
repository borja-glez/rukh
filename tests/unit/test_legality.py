"""Tests for rukh.eval.legality: sampling validation prefixes and the unmasked legality rate."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import chess
import pytest
import torch

from rukh.eval.legality import (
    Position,
    board_of,
    header,
    legality,
    position_at,
    sample_positions,
)
from rukh.infer import SampleConfig
from rukh.models import DecoderConfig, MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

pytestmark = pytest.mark.unit

# ``rukh.eval`` re-exports the function under the module's own name, so the module object for
# monkeypatching has to be asked for explicitly.
legality_module = importlib.import_module("rukh.eval.legality")

TOY = DecoderConfig(vocab_size=2030, n_layer=2, n_head=2, d_model=32, block=64)
GAME = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6"


@pytest.fixture(scope="module")
def tok() -> UciTokenizer:
    return UciTokenizer()


@pytest.fixture(scope="module")
def model() -> MoveDecoder:
    torch.manual_seed(0)
    return MoveDecoder(TOY).eval()


@pytest.fixture
def games(tmp_path: Path) -> Path:
    import polars as pl

    path = tmp_path / "games.parquet"
    pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "uci": [GAME, GAME, "e2e4"],
            "white_elo": [1850, 2150, 1900],
            "black_elo": [1950, 2050, 1900],
            "n_plies": [8, 8, 1],
        }
    ).write_parquet(path)
    return path


def test_position_at_keeps_the_prefix_the_history_and_the_human_move(tok: UciTokenizer) -> None:
    moves = GAME.split()
    position = position_at(tok, 7, moves, 4, 1850, 1950)
    assert position.moves == moves[:4]
    assert position.target == moves[4]
    assert position.history[:3] == header(tok, 1850, 1950)
    assert len(position.history) == 3 + 4
    assert position.elo == 1850  # ply 4 is White's turn


def test_the_elo_of_a_position_is_the_side_to_move(tok: UciTokenizer) -> None:
    moves = GAME.split()
    assert position_at(tok, 1, moves, 3, 1850, 1950).elo == 1950


def test_the_crop_keeps_the_header_and_drops_the_oldest_moves(tok: UciTokenizer) -> None:
    moves = GAME.split()
    position = position_at(tok, 1, moves, 8, 1850, 1950, block=5)
    assert len(position.history) == 5
    # The Elo conditioning survives: a plain ``history[-block:]`` would have thrown it away.
    assert position.history[:3] == header(tok, 1850, 1950)
    assert position.history[3:] == [tok.vocab[uci] for uci in moves[6:8]]


def test_a_short_history_is_left_alone(tok: UciTokenizer) -> None:
    position = position_at(tok, 1, GAME.split(), 4, 1850, 1950, block=200)
    assert len(position.history) == 7
    assert position.history[:3] == header(tok, 1850, 1950)


def test_legality_is_measured_twice_with_different_definitions(
    tok: UciTokenizer, model: MoveDecoder
) -> None:
    positions = [position_at(tok, 1, GAME.split(), ply, 1850, 1950) for ply in range(1, 6)]
    cfg = SampleConfig(temperature=0.6, top_k=20, seed=0)
    argmax = legality(model, tok, positions, cfg, mode="argmax")
    sampled = legality(model, tok, positions, cfg, mode="sampled")
    assert argmax.mode == "argmax" and argmax.temperature is None and argmax.top_k is None
    assert sampled.mode == "sampled" and sampled.temperature == 0.6 and sampled.top_k == 20
    # argmax is deterministic: the same call twice gives the same rate.
    assert legality(model, tok, positions, cfg, mode="argmax").legal == argmax.legal


def test_an_unknown_legality_mode_is_an_error(tok: UciTokenizer, model: MoveDecoder) -> None:
    with pytest.raises(ValueError, match="unknown legality mode"):
        legality(model, tok, [], mode="greedy")


def test_board_of_replays_the_prefix(tok: UciTokenizer) -> None:
    position = position_at(tok, 1, GAME.split(), 4, 1850, 1950)
    board = board_of(position)
    assert board.turn == chess.WHITE
    assert board.fullmove_number == 3


def test_sample_positions_returns_one_prefix_per_game(tok: UciTokenizer, games: Path) -> None:
    positions = sample_positions(games, 10, tok, seed=0)
    assert len(positions) == 2  # the one-move game is too short to sample from
    for position in positions:
        assert 1 <= position.ply < 8
        assert position.target == GAME.split()[position.ply]
        board_of(position)


def test_sample_positions_is_deterministic(tok: UciTokenizer, games: Path) -> None:
    first = sample_positions(games, 10, tok, seed=3)
    second = sample_positions(games, 10, tok, seed=3)
    assert [p.model_dump() for p in first] == [p.model_dump() for p in second]


def test_sample_positions_stops_at_n(tok: UciTokenizer, games: Path) -> None:
    assert len(sample_positions(games, 1, tok, seed=0)) == 1


def test_legality_counts_the_legal_proposals(
    tok: UciTokenizer, model: MoveDecoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter([True, False, True, True])

    def fake_pick(*_args: Any, **_kwargs: Any) -> tuple[None, dict[str, Any]]:
        return None, {"legal": next(answers), "raw_token": "e2e4", "top5": [], "masked": False}

    monkeypatch.setattr(legality_module, "pick_move", fake_pick)
    positions = [position_at(tok, 1, GAME.split(), ply, 1850, 1950) for ply in range(1, 5)]
    result = legality(model, tok, positions)
    assert (result.positions, result.legal) == (4, 3)
    assert result.rate == pytest.approx(0.75)


def test_legality_asks_the_model_without_the_mask(
    tok: UciTokenizer, model: MoveDecoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []

    def fake_pick(_model, _tok, _board, _history, cfg, _generator=None):  # type: ignore[no-untyped-def]
        seen.append(cfg.mask_illegal)
        return None, {"legal": True, "raw_token": "e2e4", "top5": [], "masked": False}

    monkeypatch.setattr(legality_module, "pick_move", fake_pick)
    legality(model, tok, [position_at(tok, 1, GAME.split(), 2, 1850, 1950)])
    assert seen == [False]


def test_legality_of_a_real_toy_model_is_a_rate(tok: UciTokenizer, model: MoveDecoder) -> None:
    positions = [position_at(tok, 1, GAME.split(), ply, 1850, 1950) for ply in range(1, 6)]
    result = legality(model, tok, positions)
    assert result.positions == 5
    assert 0.0 <= result.rate <= 1.0
    assert result.legal == int(result.rate * result.positions)


def test_legality_without_positions_is_zero(tok: UciTokenizer, model: MoveDecoder) -> None:
    empty: list[Position] = []
    assert legality(model, tok, empty).rate == 0.0
