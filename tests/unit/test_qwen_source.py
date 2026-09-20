"""Tests for rukh.eval.qwen_source: the three ways a text model can fail where the decoder cannot.

Nothing here loads a language model. What is worth testing is the parsing and the accounting: the
decoder's vocabulary *is* the set of moves, so its only failure is "legal move, wrong position",
while a model writing SAN can produce something that is not a move, a move this position does not
allow, or a move two pieces could make. Folding those into one number would throw away the most
interesting half of the comparison.
"""

from __future__ import annotations

import chess
import pytest

from rukh.eval.qwen_source import QwenPlayer, QwenStats, parse_san_answer

pytestmark = pytest.mark.unit


def _board(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (" e4", "e2e4"),
        ("e4 e5 2. Nf3", "e2e4"),  # it kept going; only the answer to the question counts
        ("1. e4", "e2e4"),  # a move number in front is bookkeeping
        ("e4!?", "e2e4"),  # commentary glyphs are not part of the move
        ("  Nf3  ", "g1f3"),
    ],
    ids=["plain", "kept-going", "numbered", "annotated", "padded"],
)
def test_a_legal_answer_is_read_as_the_move_it_names(answer: str, expected: str) -> None:
    move, outcome = parse_san_answer(chess.Board(), answer)
    assert outcome == "legal"
    assert move is not None and move.uci() == expected


def test_a_move_number_alone_reaches_for_the_next_token() -> None:
    move, outcome = parse_san_answer(chess.Board(), "1. e4")
    assert outcome == "legal" and move is not None


def test_something_that_is_not_a_move_is_unparseable() -> None:
    """`Nf9` is not a square; no position in chess allows it, so it is its own failure."""
    for answer in ("Nf9", "hola", "O-O-O-O", "-"):
        _, outcome = parse_san_answer(chess.Board(), answer)
        assert outcome == "unparseable", answer


def test_a_well_formed_move_this_position_forbids_is_illegal() -> None:
    """The only failure the decoder can have, so it is the one that makes the two comparable."""
    move, outcome = parse_san_answer(chess.Board(), "Qh5")  # the queen is boxed in at the start
    assert outcome == "illegal" and move is None


def test_san_two_pieces_could_satisfy_is_ambiguous() -> None:
    """SAN is contextual: this is the failure mode the decoder's vocabulary cannot even express."""
    board = _board(["g1f3", "d7d5", "d2d4", "g8f6"])
    assert board.san(chess.Move.from_uci("b1d2")) == "Nbd2"  # both knights reach d2
    move, outcome = parse_san_answer(board, "Nd2")
    assert outcome == "ambiguous" and move is None


def test_an_empty_answer_is_counted_as_such() -> None:
    assert parse_san_answer(chess.Board(), "   ")[1] == "empty"
    assert parse_san_answer(chess.Board(), "")[1] == "empty"


def test_the_stats_add_up_and_the_rate_is_of_what_was_asked() -> None:
    stats = QwenStats()
    for outcome in ("legal", "legal", "illegal", "unparseable", "ambiguous", "empty"):
        stats.record(outcome)
    assert stats.asked == 6
    assert (stats.legal, stats.illegal, stats.unparseable, stats.ambiguous, stats.empty) == (
        2,
        1,
        1,
        1,
        1,
    )
    assert stats.legal_rate == pytest.approx(2 / 6)


class _Scripted:
    """A stand-in for the language model that answers from a list, in order."""

    device = "cpu"

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)


class _Player(QwenPlayer):
    def __init__(self, answers: list[str]) -> None:
        super().__init__(_Scripted(answers), tokenizer=None)
        self._answers = list(answers)

    def answer(self, board: chess.Board) -> str:  # noqa: ARG002
        return self._answers.pop(0) if self._answers else ""


def test_a_failed_proposal_is_rescued_and_still_counted() -> None:
    """A game has to finish, and the rescue must not hide that the model's own answer failed.

    Same rule as the decoder's `illegal_proposals`: the fallback keeps the harness running, the
    count is what the legality rate is built from.
    """
    player = _Player(["Nf9", "e4"])
    move, was_legal = player.choose(chess.Board())
    assert was_legal is False
    assert move in chess.Board().legal_moves  # rescued, so the game continues
    assert player.stats.unparseable == 1

    move, was_legal = player.choose(chess.Board())
    assert was_legal is True
    assert move is not None and move.uci() == "e2e4"
    assert player.stats.legal == 1


def test_the_rescue_does_not_ask_the_model_again() -> None:
    """Asking until it answers legally would measure a different model from the reported one."""
    player = _Player(["Nf9"])
    player.choose(chess.Board())
    assert player.stats.asked == 1


def test_a_finished_position_rescues_to_nothing_rather_than_crashing() -> None:
    mate = chess.Board("7k/5KQ1/8/8/8/8/8/8 b - - 0 1")
    move, was_legal = _Player(["Kh7"]).choose(mate)
    assert move is None and was_legal is False
