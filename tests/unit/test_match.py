"""The head-to-head instrument, checked before anything is measured with it.

Its whole reason to exist is that subtracting two noisy absolutes is the wrong way to get a
difference, so the first thing it has to prove is that *it* is not biased: the same model on both
sides must come out at zero, with an interval that contains zero.
"""

from __future__ import annotations

import random

import chess
import pytest

from rukh.eval.match import (
    MatchGame,
    elo_difference,
    games_for_edge,
    opening_book,
    play_match,
    summarise,
)

pytestmark = pytest.mark.unit


class _Scripted:
    """A ``Player`` that plays a fixed policy: the cheapest way to test the harness itself."""

    def __init__(self, seed: int = 0, prefer_first: bool = False) -> None:
        self.rng = random.Random(seed)
        self.prefer_first = prefer_first
        self.limit = 200
        self.started: list[tuple[int, int]] = []

    def start(self, white_elo: int, black_elo: int) -> None:
        self.started.append((white_elo, black_elo))

    def choose(self, board: chess.Board) -> tuple[chess.Move | None, bool]:
        moves = sorted(board.legal_moves, key=lambda move: move.uci())
        if self.prefer_first:
            return moves[0], True
        return self.rng.choice(moves), True

    def observe(self, move: chess.Move) -> None:
        pass


def test_the_elo_difference_is_the_logit_of_the_score():
    assert elo_difference(0.5) == 0.0
    assert elo_difference(0.5715) == pytest.approx(50.0, abs=0.5)
    assert elo_difference(0.75) == pytest.approx(190.85, abs=0.1)
    # A clean sweep has no finite fit, so it is capped with its sign rather than returning inf.
    assert elo_difference(1.0) == 1200.0
    assert elo_difference(0.0) == -1200.0


def test_the_arithmetic_that_decides_whether_to_pay_for_a_match():
    """Eight times cheaper than the ladder, and the number that says so."""
    assert games_for_edge(50) == 185
    # Halving the edge roughly quadruples the price, which is the whole shape of the problem.
    assert games_for_edge(25) > 3.5 * games_for_edge(50)
    assert games_for_edge(100) < games_for_edge(50)
    assert games_for_edge(0) == 0


def test_an_odd_number_of_games_is_refused_instead_of_silently_unbalanced():
    """Mirroring is the point; an odd match gives one model an extra game as white."""
    a, b = _Scripted(1), _Scripted(2)
    with pytest.raises(ValueError, match="even number of games"):
        play_match(a, b, games=7)


def test_every_opening_is_played_twice_with_the_colours_swapped():
    played = play_match(_Scripted(1), _Scripted(2), games=6, opening_plies=4, seed=3)
    assert len(played) == 6
    by_opening: dict[str, list[bool]] = {}
    for game in played:
        by_opening.setdefault(game.opening, []).append(game.a_white)
    assert len(by_opening) == 3
    for colours in by_opening.values():
        assert sorted(colours) == [False, True], "each opening must be played from both sides"


def test_the_book_is_seeded_and_its_openings_are_legal():
    first = opening_book(4, plies=6, seed=11)
    assert first == opening_book(4, plies=6, seed=11)
    assert opening_book(4, plies=6, seed=12) != first
    for opening in first:
        board = chess.Board()
        for uci in opening:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            board.push(move)
        assert not board.is_game_over()


def test_the_same_model_on_both_sides_measures_zero():
    """The control the whole instrument rests on: no difference must read as no difference."""
    played = play_match(_Scripted(5), _Scripted(5), games=20, opening_plies=4, seed=7)
    result = summarise(played, names=("x", "x"))
    assert result.games == 20
    assert result.wins + result.draws + result.losses == 20
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low <= 0 <= result.ci_high, "an unbiased instrument includes zero"
    assert result.separated is False


def test_a_one_sided_match_separates_and_says_so():
    scores = [1.0] * 30 + [0.5] * 2
    played = [
        MatchGame(
            index=i,
            opening="e2e4",
            a_white=bool(i % 2),
            result="1-0",
            score=score,
            plies=40,
            a_illegal=0,
            b_illegal=0,
        )
        for i, score in enumerate(scores)
    ]
    result = summarise(played)
    assert result.elo > 300
    assert result.separated is True
    assert result.ci_low is not None and result.ci_low > 0


def test_summarising_nothing_is_refused():
    with pytest.raises(ValueError, match="no games"):
        summarise([])


def test_the_illegal_proposals_of_both_sides_are_counted_separately():
    """DPO is known to trade legality for margin (D-071), so a match has to watch both models."""
    played = [
        MatchGame(
            index=0,
            opening="e2e4",
            a_white=True,
            result="1-0",
            score=1.0,
            plies=10,
            a_illegal=3,
            b_illegal=7,
        )
    ]
    result = summarise(played)
    assert (result.a_illegal, result.b_illegal) == (3, 7)


class _Counting:
    """A player that records everything it is told, to prove the loop tells it."""

    def __init__(self) -> None:
        self.limit = 200
        self.seen: list[str] = []
        self.starts = 0

    def start(self, white_elo: int, black_elo: int) -> None:
        self.starts += 1
        self.seen = []

    def choose(self, board: chess.Board) -> tuple[chess.Move | None, bool]:
        return next(iter(board.legal_moves)), True

    def observe(self, move: chess.Move) -> None:
        self.seen.append(move.uci())


class _CountingOpponent(_Counting):
    """The same thing on the other side of the board: an ``Opponent`` returns a move, not a pair.

    The two protocols differ for a reason -- a ``Player`` has to say whether its own proposal was
    legal and an opponent never proposes anything illegal -- and `_PlayerOpponent` in the match
    harness is exactly this adapter for a real model.
    """

    def choose(self, board: chess.Board) -> chess.Move:  # type: ignore[override]
        return next(iter(board.legal_moves))


def test_a_board_handed_in_with_moves_is_replayed_into_both_sides():
    """The first bug the control found: a match from an opening book played it blind.

    `play_game_with` used to start the player and then begin at whatever position the board was
    in, without telling it how the game got there. The prompt was a game that never happened, and
    the model answered with illegal moves -- which read as a broken model rather than a broken
    harness.
    """
    from rukh.infer.game import play_game_with

    opening = ["e2e4", "e7e5", "g1f3"]
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)

    player, opponent = _Counting(), _CountingOpponent()
    play_game_with(player, opponent, max_plies=len(opening) + 4, board=board)
    assert player.seen[: len(opening)] == opening
    assert opponent.seen[: len(opening)] == opening


def test_a_stateful_opponent_is_shown_the_moves_too():
    """The second bug: only the player was told, so the other model played from move one forever."""
    from rukh.infer.game import play_game_with

    player, opponent = _Counting(), _CountingOpponent()
    play_game_with(player, opponent, max_plies=6)
    assert len(opponent.seen) == len(player.seen) > 0
    assert opponent.seen == player.seen


def test_an_opponent_without_observe_is_left_alone():
    """Stockfish and the random mover have no `observe`; the loop must not invent one."""
    from rukh.infer.game import RandomOpponent, play_game_with

    player = _Counting()
    result = play_game_with(player, RandomOpponent(seed=3), max_plies=8)
    assert result.plies > 0


def test_the_report_of_a_match_with_no_settings_says_so_instead_of_crashing() -> None:
    """Two artefacts predate the settings block; reading one must not be an AttributeError."""
    from rukh.eval.match import MatchResult, render_markdown

    result = MatchResult(a="old", b="older", games=2, score=0.5, wins=1, draws=0, losses=1, elo=0.0)
    assert result.settings is None
    text = render_markdown(result, [])
    assert "not recorded" in text
