"""Tests for rukh.infer: legality masking, temperature, promotions and whole games."""

from __future__ import annotations

import chess
import pytest
import torch

from rukh.infer import (
    GameResult,
    RandomOpponent,
    SampleConfig,
    StockfishOpponent,
    legal_token_ids,
    legality_rate,
    pick_move,
    play_game,
    result_token,
)
from rukh.models import DecoderConfig, MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

# The Stockfish game carries only `engine`, so `-m unit` never needs the binary.
unit = pytest.mark.unit

TOY = DecoderConfig(vocab_size=2030, n_layer=2, n_head=2, d_model=32, block=64)
PROMOTION_FEN = "8/P6k/8/8/8/8/6K1/8 w - - 0 1"


@pytest.fixture(scope="module")
def tok() -> UciTokenizer:
    return UciTokenizer()


@pytest.fixture(scope="module")
def model() -> MoveDecoder:
    torch.manual_seed(0)
    return MoveDecoder(TOY).eval()


def history(tok: UciTokenizer) -> list[int]:
    return [tok.bos_id, tok.vocab["<w1800>"], tok.vocab["<b1800>"]]


@unit
def test_the_initial_position_has_twenty_legal_tokens(tok: UciTokenizer) -> None:
    board = chess.Board()
    ids = legal_token_ids(board, tok)
    assert len(ids) == 20
    assert ids == sorted(set(ids))
    for token_id in ids:
        assert board.is_legal(chess.Move.from_uci(tok.ids[token_id]))
    assert {tok.ids[i] for i in ids} == {move.uci() for move in board.legal_moves}


@unit
def test_castling_is_the_two_square_king_move(tok: UciTokenizer) -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    tokens = {tok.ids[i] for i in legal_token_ids(board, tok)}
    assert {"e1g1", "e1c1"} <= tokens


@unit
def test_masked_sampling_never_proposes_an_illegal_move(
    model: MoveDecoder, tok: UciTokenizer
) -> None:
    board = chess.Board()
    cfg = SampleConfig(temperature=1.5, top_k=None, mask_illegal=True, seed=0)
    generator = cfg.generator()
    seen = set()
    for _ in range(1000):
        move, report = pick_move(model, tok, board, history(tok), cfg, generator)
        assert move is not None
        assert report["legal"] is True and report["masked"] is True
        assert board.is_legal(move)
        seen.add(move.uci())
    assert len(seen) > 1  # the mask does not collapse the distribution


@unit
def test_without_the_mask_an_illegal_token_is_reported_as_such(
    model: MoveDecoder, tok: UciTokenizer
) -> None:
    board = chess.Board()
    cfg = SampleConfig(temperature=1.0, top_k=None, mask_illegal=False, seed=1)
    generator = cfg.generator()
    illegal = 0
    for _ in range(50):
        move, report = pick_move(model, tok, board, history(tok), cfg, generator)
        assert report["masked"] is False
        assert report["raw_token"] in tok.vocab
        if move is None:
            illegal += 1
            assert report["legal"] is False
    assert illegal > 0  # an untrained model is almost never legal on its own


@unit
def test_zero_temperature_is_argmax(model: MoveDecoder, tok: UciTokenizer) -> None:
    board = chess.Board()
    cfg = SampleConfig(temperature=0.0, top_k=None, mask_illegal=True)
    ids = legal_token_ids(board, tok)
    idx = torch.tensor([history(tok)], dtype=torch.long)
    logits = model.next_logits(idx)[0]
    best = tok.ids[ids[int(torch.argmax(logits[torch.tensor(ids)]))]]
    for _ in range(5):
        move, report = pick_move(model, tok, board, history(tok), cfg)
        assert move is not None and move.uci() == best
        assert report["raw_token"] == best


@unit
def test_top_k_restricts_the_reported_distribution(model: MoveDecoder, tok: UciTokenizer) -> None:
    board = chess.Board()
    cfg = SampleConfig(temperature=1.0, top_k=3, mask_illegal=True, seed=2)
    drawn = set()
    generator = cfg.generator()
    for _ in range(200):
        move, report = pick_move(model, tok, board, history(tok), cfg, generator)
        assert move is not None
        drawn.add(move.uci())
        assert len(report["top5"]) <= 3
        assert sum(probability for _, probability in report["top5"]) == pytest.approx(1.0, abs=1e-5)
    assert len(drawn) <= 3


@unit
def test_a_forced_promotion_yields_a_promotion_token(model: MoveDecoder, tok: UciTokenizer) -> None:
    board = chess.Board(PROMOTION_FEN)
    tokens = {tok.ids[i] for i in legal_token_ids(board, tok)}
    assert {"a7a8q", "a7a8r", "a7a8b", "a7a8n"} <= tokens
    cfg = SampleConfig(temperature=1.0, top_k=None, mask_illegal=True, seed=3)
    generator = cfg.generator()
    for _ in range(20):
        move, report = pick_move(model, tok, board, history(tok), cfg, generator)
        assert move is not None
        if move.from_square == chess.A7:
            assert move.promotion is not None
            assert len(report["raw_token"]) == 5


@unit
def test_a_finished_position_has_no_legal_tokens(model: MoveDecoder, tok: UciTokenizer) -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # checkmate
    assert legal_token_ids(board, tok) == []
    move, report = pick_move(model, tok, board, history(tok), SampleConfig(seed=0))
    assert move is None and report["legal"] is False and report["top5"] == []


@unit
def test_a_full_game_against_a_random_opponent(model: MoveDecoder, tok: UciTokenizer) -> None:
    cfg = SampleConfig(temperature=1.0, top_k=20, mask_illegal=True, seed=7)
    result = play_game(model, tok, RandomOpponent(seed=7), cfg, max_plies=40)
    assert isinstance(result, GameResult)
    assert result.result in ("1-0", "0-1", "1/2-1/2", "*")
    assert result.plies == len(result.moves)
    assert result.illegal_proposals == 0  # the mask makes them impossible
    assert result.termination

    board = chess.Board()
    for uci in result.moves:
        move = chess.Move.from_uci(uci)
        assert board.is_legal(move)
        board.push(move)


@unit
def test_an_unmasked_game_counts_illegal_proposals_and_still_ends(
    model: MoveDecoder, tok: UciTokenizer
) -> None:
    cfg = SampleConfig(temperature=1.0, top_k=None, mask_illegal=False, seed=5)
    result = play_game(
        model, tok, RandomOpponent(seed=5), cfg, model_color=chess.BLACK, max_plies=20
    )
    assert result.illegal_proposals > 0
    assert result.plies == 20 or result.result != "*"


@unit
def test_the_game_stops_at_the_block_size_by_default(model: MoveDecoder, tok: UciTokenizer) -> None:
    cfg = SampleConfig(temperature=1.0, top_k=20, seed=11)
    result = play_game(model, tok, RandomOpponent(seed=11), cfg)
    assert result.plies <= TOY.block - 4


@unit
def test_the_same_seed_replays_the_same_game(model: MoveDecoder, tok: UciTokenizer) -> None:
    cfg = SampleConfig(temperature=1.0, top_k=20, seed=13)
    first = play_game(model, tok, RandomOpponent(seed=13), cfg, max_plies=30)
    second = play_game(model, tok, RandomOpponent(seed=13), cfg, max_plies=30)
    assert first.moves == second.moves


@unit
def test_legality_rate_is_a_share(model: MoveDecoder, tok: UciTokenizer) -> None:
    boards = [chess.Board() for _ in range(5)]
    histories = [history(tok) for _ in boards]
    rate = legality_rate(model, tok, boards, histories, SampleConfig(seed=0))
    assert 0.0 <= rate <= 1.0
    assert legality_rate(model, tok, [], [], SampleConfig()) == 0.0


@unit
def test_result_tokens_map_to_the_vocabulary(tok: UciTokenizer) -> None:
    assert result_token(tok, "1-0") == tok.vocab["<1-0>"]
    assert result_token(tok, "1/2-1/2") == tok.vocab["<1/2>"]
    assert result_token(tok, "*") is None


@unit
def test_sample_config_rejects_impossible_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SampleConfig(temperature=-0.1)
    with pytest.raises(ValueError, match="top_k"):
        SampleConfig(top_k=0)
    with pytest.raises(ValueError):
        SampleConfig(temp=0.5)  # type: ignore[call-arg]
    assert SampleConfig(seed=None).generator() is None


@unit
def test_cli_play_is_registered() -> None:
    from typer.testing import CliRunner

    from rukh.cli import app

    result = CliRunner().invoke(app, ["play", "--help"])
    assert result.exit_code == 0, result.output
    for option in ("--ckpt", "--games", "--opponent", "--no-mask"):
        assert option in result.output


@pytest.mark.engine
def test_a_game_against_stockfish(model: MoveDecoder, tok: UciTokenizer) -> None:
    from rukh.engine import find_stockfish

    if find_stockfish() is None:
        pytest.skip("Stockfish is not available")
    cfg = SampleConfig(temperature=1.0, top_k=20, seed=17)
    with StockfishOpponent(elo=1400, move_time=0.02) as rival:
        result = play_game(model, tok, rival, cfg, model_color=chess.WHITE, max_plies=20)
    assert result.plies > 0 and result.illegal_proposals == 0
