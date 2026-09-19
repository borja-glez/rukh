"""Tests for rukh.models.squares: the 69-token FEN scheme, its vocabulary and the round trip."""

from __future__ import annotations

import chess
import pytest

from rukh.models.squares import (
    CLOCK_EDGES,
    SQUARE_IDS,
    SQUARE_TOKENS,
    SQUARE_VOCAB,
    SQUARE_VOCAB_SIZE,
    castling_strings,
    clock_bucket,
    fen_to_tokens,
    square_name,
    tokens_to_fen,
    tokens_to_strings,
    vocab_hash,
)
from rukh.tokenize.uci_vocab import squares

pytestmark = pytest.mark.unit

# Pinned like the UCI vocabulary: an encoder is only meaningful with the vocabulary it saw.
SQUARE_VOCAB_SHA256 = "b0f530895c035c07423dfa246ad5e874c7406ffbaa1b2155601ebf09e89c98e9"
OPENING = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]


def played(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return board


def test_the_vocabulary_is_stable() -> None:
    assert vocab_hash() == SQUARE_VOCAB_SHA256
    assert len(SQUARE_VOCAB) == SQUARE_VOCAB_SIZE == 47
    assert len(set(SQUARE_VOCAB)) == SQUARE_VOCAB_SIZE
    assert SQUARE_VOCAB[:4] == ["<pad>", "<mask>", "<cls>", "<empty>"]
    assert SQUARE_VOCAB[4:16] == list("PNBRQKpnbrqk")
    assert len(castling_strings()) == 16
    assert castling_strings()[0] == "-" and castling_strings()[-1] == "KQkq"


def test_a_position_is_exactly_69_tokens() -> None:
    board = played(OPENING)
    assert len(fen_to_tokens(board.fen())) == SQUARE_TOKENS == 69
    fen4 = " ".join(board.fen().split()[:4])
    assert len(fen_to_tokens(fen4)) == SQUARE_TOKENS  # the P1 four-field FEN is accepted too


def test_the_layout_is_cls_board_turn_castling_en_passant_clock() -> None:
    names = tokens_to_strings(fen_to_tokens(chess.Board().fen()))
    assert names[0] == "<cls>"
    assert names[65:] == ["turn:w", "castle:KQkq", "ep:none", "clock:0"]
    assert names[1] == "R" and names[2] == "P" and names[3] == "<empty>"  # a1, a2, a3


def test_every_square_matches_python_chess() -> None:
    board = chess.Board()
    for move in OPENING:
        board.push_uci(move)
        names = tokens_to_strings(fen_to_tokens(board.fen()))[1:65]
        for index, name in enumerate(names):
            piece = board.piece_at(chess.square(index // 8, index % 8))
            assert name == (piece.symbol() if piece else "<empty>"), square_name(index)


def test_the_64_squares_round_trip_through_the_four_field_fen() -> None:
    board = chess.Board()
    for move in [*OPENING, "f1e2", "c8d7", "e1g1"]:
        board.push_uci(move)
        fen4 = " ".join(board.fen().split()[:4])
        assert tokens_to_fen(fen_to_tokens(fen4)) == fen4
        assert tokens_to_fen(fen_to_tokens(board.fen())) == fen4


def test_castling_rights_are_distinguished() -> None:
    board = played(OPENING)
    fen4 = " ".join(board.fen().split()[:4])
    fields = fen4.split()
    variants = {
        rights: fen_to_tokens(f"{fields[0]} {fields[1]} {rights} {fields[3]}")[66]
        for rights in ("KQkq", "KQk", "Kq", "-")
    }
    assert len(set(variants.values())) == 4
    assert variants["-"] == SQUARE_IDS["castle:-"]
    # The order inside the field does not matter, the rights do.
    assert fen_to_tokens(f"{fields[0]} {fields[1]} qK {fields[3]}")[66] == variants["Kq"]
    with pytest.raises(ValueError, match="unsupported castling field"):
        fen_to_tokens(f"{fields[0]} {fields[1]} AHah {fields[3]}")


def test_en_passant_is_distinguished_and_does_not_disturb_the_board() -> None:
    board = played(["e2e4", "a7a6", "e4e5", "d7d5"])  # d6 is a legal en-passant target
    with_ep = fen_to_tokens(board.fen())
    without = fen_to_tokens(board.fen().replace(" d6 ", " - "))
    assert with_ep[67] == SQUARE_IDS["ep:d"] and without[67] == SQUARE_IDS["ep:none"]
    assert with_ep[:67] == without[:67]  # only the en-passant slot changes
    assert tokens_to_fen(with_ep).split()[3] == "d6"
    assert fen_to_tokens("8/8/8/8/8/8/8/8 b - c3")[67] == SQUARE_IDS["ep:c"]
    assert tokens_to_fen(fen_to_tokens("8/8/8/8/8/8/8/8 b - c3")).split()[3] == "c3"
    with pytest.raises(ValueError, match="en-passant square"):
        fen_to_tokens("8/8/8/8/8/8/8/8 w - e4")


def test_the_halfmove_clock_is_bucketed() -> None:
    assert CLOCK_EDGES == (6, 25, 50)
    assert [clock_bucket(n) for n in (0, 5, 6, 24, 25, 49, 50, 120)] == [0, 0, 1, 1, 2, 2, 3, 3]
    start = chess.Board().fen().split()
    fen = " ".join(start[:4])
    assert fen_to_tokens(f"{fen} 0 1")[68] == SQUARE_IDS["clock:0"]
    assert fen_to_tokens(f"{fen} 30 40")[68] == SQUARE_IDS["clock:2"]
    assert fen_to_tokens(fen)[68] == SQUARE_IDS["clock:0"]  # a fen4 carries no counter
    with pytest.raises(ValueError, match="cannot be negative"):
        clock_bucket(-1)


def test_the_square_order_is_the_uci_vocabulary_order() -> None:
    assert square_name(0) == "a1" and square_name(63) == "h8"
    assert [square_name(i) for i in range(64)] == squares()


def test_a_malformed_fen_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 4 fields"):
        fen_to_tokens("8/8/8/8/8/8/8/8 w")
    with pytest.raises(ValueError, match="8 ranks"):
        fen_to_tokens("8/8/8 w - -")
    with pytest.raises(ValueError, match="covers 7 files"):
        fen_to_tokens("7/8/8/8/8/8/8/8 w - -")
    with pytest.raises(ValueError, match="unknown piece"):
        fen_to_tokens("xxxxxxxx/8/8/8/8/8/8/8 w - -")
    with pytest.raises(ValueError, match="side to move"):
        fen_to_tokens("8/8/8/8/8/8/8/8 x - -")
    with pytest.raises(ValueError, match="69 tokens"):
        tokens_to_fen([1, 2, 3])
