"""Two models playing each other, which is how a *difference* in strength gets measured.

Everything published so far measures a model against the Stockfish ladder and reports an absolute
Elo. That is the right instrument for "how strong is this" and the wrong one for "is this stronger
than that", and P5's acceptance criterion asks the second question: **+50 Elo over its base**.

The arithmetic says how wrong. A 50-Elo edge is a 0.5715 expected score head to head, and
separating 0.5715 from 0.5 at 95 % takes **185 games**. The same 50 Elo read off the ladder is
0.0712 of score around a base of 0.45, and separating two such estimates takes **757 games per
side**, 1 514 in total -- before adding the ladder's own irreproducibility, which P4 measured at
about 40 Elo of one sigma (D-107) because Stockfish plays on a clock and randomises on purpose.

Eight times cheaper and one source of noise fewer, for the simple reason that both models play
*the same game*: there is no third party whose mood has to be averaged out.

Two details make the number honest.

**Colours are mirrored.** Every opening is played twice, once with each model on white, so a
repertoire that happens to suit one side cannot decide the match. A match with an odd number of
games is refused rather than silently unbalanced.

**The openings are shared and seeded.** Both games of a pair start from the same position, so the
pair is a comparison and not two samples.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import chess
import numpy as np
from pydantic import BaseModel, ConfigDict

from rukh.infer.game import DecoderPlayer, play_game_with

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from rukh.infer.game import Player
    from rukh.infer.sampler import SampleConfig
    from rukh.models import MoveDecoder
    from rukh.tokenize.uci_vocab import UciTokenizer

__all__ = [
    "MatchGame",
    "MatchResult",
    "elo_difference",
    "games_for_edge",
    "opening_book",
    "play_match",
    "render_markdown",
    "summarise",
    "write_match",
]


class MatchGame(BaseModel):
    """One game of the match, from ``a``'s point of view."""

    model_config = ConfigDict(extra="forbid")

    index: int
    opening: str
    """The moves both games of this pair started from, in UCI."""
    a_white: bool
    result: str
    score: float
    """1 for a win by ``a``, 0.5 for a draw, 0 for a loss."""
    plies: int
    a_illegal: int
    b_illegal: int


class MatchResult(BaseModel):
    """What the match measured: a score, an Elo difference and its interval."""

    model_config = ConfigDict(extra="forbid")

    a: str
    b: str
    games: int
    score: float
    """``a``'s share of the points. 0.5 means the match found no difference."""
    wins: int
    draws: int
    losses: int
    elo: float
    """``400 log10(p / (1 - p))``: how much stronger ``a`` is than ``b``, in Elo."""
    ci_low: float | None = None
    ci_high: float | None = None
    separated: bool = False
    """Whether the interval excludes zero, which is the claim the criterion is read on."""
    a_illegal: int = 0
    b_illegal: int = 0

    @property
    def decisive(self) -> int:
        return self.wins + self.losses


def elo_difference(score: float, cap: float = 1200.0) -> float:
    """``400 log10(p / (1 - p))``, clamped when one side won everything.

    A clean sweep has no finite fit -- the logit runs off -- so it is reported at the cap with the
    sign it earned, and ``MatchResult.separated`` is what a reader should look at instead.
    """
    if score <= 0.0:
        return -cap
    if score >= 1.0:
        return cap
    return max(-cap, min(cap, 400.0 * math.log10(score / (1.0 - score))))


def games_for_edge(elo: float, level: float = 0.95) -> int:
    """How many games it takes to tell an edge of ``elo`` from nothing.

    The inverse of the question every match asks, and the reason this module exists: run it
    **before** paying for the games, not after looking at them.
    """
    if elo == 0:
        return 0
    from scipy.stats import norm  # noqa: PLC0415 - only needed here

    p = 1.0 / (1.0 + 10.0 ** (-abs(elo) / 400.0))
    z = float(norm.ppf(0.5 + level / 2.0))
    return math.ceil(z * z * p * (1.0 - p) / (p - 0.5) ** 2)


def opening_book(pairs: int, plies: int = 6, seed: int = 0) -> list[list[str]]:
    """``pairs`` random legal openings, each to be played twice with the colours swapped.

    Random rather than a curated book on purpose: a book of "sound" openings would measure the two
    models inside somebody else's taste, and the question here is which of the two plays better
    chess, not which of them knows the Ruy Lopez.
    """
    rng = random.Random(seed)
    book: list[list[str]] = []
    while len(book) < pairs:
        board = chess.Board()
        moves: list[str] = []
        for _ in range(plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            moves.append(move.uci())
            board.push(move)
        if len(moves) == plies and not board.is_game_over():
            book.append(moves)
    return book


def _board_from(opening: Sequence[str]) -> chess.Board:
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    return board


class _PlayerOpponent:
    """A ``Player`` seen as an ``Opponent``: what lets two decoders face each other.

    ``play_game_with`` drives one side through the ``Player`` protocol and asks the other for a
    move. Wrapping the second model here means the match reuses the very loop every published game
    was played on, instead of a second implementation that could disagree with it.
    """

    def __init__(self, player: Player) -> None:
        self.player = player
        self.illegal = 0

    def start(self, white_elo: int, black_elo: int) -> None:
        self.illegal = 0
        self.player.start(white_elo, black_elo)

    def choose(self, board: chess.Board) -> chess.Move:
        move, legal = self.player.choose(board)
        if not legal:
            self.illegal += 1
        if move is None:
            # The loop has no way to say "I resign"; the first legal move ends the game honestly
            # and the illegal count already recorded that the model had nothing to offer.
            return next(iter(board.legal_moves))
        return move

    def observe(self, move: chess.Move) -> None:
        self.player.observe(move)


def play_match(
    a: Player,
    b: Player,
    games: int = 200,
    opening_plies: int = 6,
    seed: int = 0,
    header_elo: int = 1800,
    max_plies: int | None = None,
) -> list[MatchGame]:
    """Play ``games`` games between two players, colours mirrored over shared openings."""
    if games < 2 or games % 2:
        raise ValueError(f"a mirrored match needs an even number of games, got {games}")
    book = opening_book(games // 2, plies=opening_plies, seed=seed)
    played: list[MatchGame] = []
    for opening in book:
        for a_white in (True, False):
            opponent = _PlayerOpponent(b)
            # `play_game_with` starts the player and replays the board's opening into both sides;
            # the opponent is started here because the loop does not know it is a model.
            opponent.start(header_elo, header_elo)
            board = _board_from(opening)
            result = play_game_with(
                a,
                opponent,
                model_color=chess.WHITE if a_white else chess.BLACK,
                white_elo=header_elo,
                black_elo=header_elo,
                max_plies=max_plies,
                board=board,
            )
            score = _score_for_a(result.result, a_white)
            played.append(
                MatchGame(
                    index=len(played),
                    opening=" ".join(opening),
                    a_white=a_white,
                    result=result.result,
                    score=score,
                    plies=result.plies,
                    a_illegal=result.illegal_proposals,
                    b_illegal=opponent.illegal,
                )
            )
    return played


def _score_for_a(result: str, a_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if a_white else 0.0
    if result == "0-1":
        return 0.0 if a_white else 1.0
    return 0.5  # a game that was cut short is a draw, the same rule the ladder uses


def summarise(
    played: Sequence[MatchGame],
    names: tuple[str, str] = ("a", "b"),
    samples: int = 2_000,
    seed: int = 0,
    level: float = 0.95,
) -> MatchResult:
    """The match as one number with its interval, bootstrapped over games."""
    if not played:
        raise ValueError("cannot summarise a match with no games")
    scores = np.array([game.score for game in played], dtype=float)
    score = float(scores.mean())
    rng = np.random.default_rng(seed)
    draws = rng.choice(scores, size=(samples, len(scores)), replace=True).mean(axis=1)
    elos = np.array([elo_difference(float(value)) for value in draws])
    low, high = (np.quantile(elos, (1 - level) / 2), np.quantile(elos, 1 - (1 - level) / 2))
    return MatchResult(
        a=names[0],
        b=names[1],
        games=len(played),
        score=score,
        wins=int(sum(1 for game in played if game.score == 1.0)),
        draws=int(sum(1 for game in played if game.score == 0.5)),
        losses=int(sum(1 for game in played if game.score == 0.0)),
        elo=elo_difference(score),
        ci_low=float(low),
        ci_high=float(high),
        separated=bool(low > 0 or high < 0),
        a_illegal=int(sum(game.a_illegal for game in played)),
        b_illegal=int(sum(game.b_illegal for game in played)),
    )


def decoder_player(model: MoveDecoder, tok: UciTokenizer, cfg: SampleConfig) -> DecoderPlayer:
    """The same player the ladder uses, so a match and a ladder run measure the same model."""
    return DecoderPlayer(model, tok, cfg)


def render_markdown(result: MatchResult, played: Sequence[MatchGame]) -> str:
    """The match as a report, with the arithmetic that says whether to believe it."""
    sign = "+" if result.elo >= 0 else ""
    interval = (
        f"{result.ci_low:.0f} to {result.ci_high:.0f}"
        if result.ci_low is not None and result.ci_high is not None
        else "n/a"
    )
    needed = games_for_edge(result.elo) if result.elo else 0
    white = [game for game in played if game.a_white]
    black = [game for game in played if not game.a_white]
    lines = [
        f"# `{result.a}` against `{result.b}`",
        "",
        "Two models playing each other over the same openings with the colours mirrored. This is",
        "the instrument for a *difference* in strength: both models play the same game, so there",
        "is no third party whose mood has to be averaged out.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Games | {result.games} |",
        f"| Score for `{result.a}` | {result.score:.4f} |",
        f"| Won / drawn / lost | {result.wins} / {result.draws} / {result.losses} |",
        f"| Elo difference | **{sign}{result.elo:.0f}** |",
        f"| 95 % CI | {interval} |",
        f"| Separated from zero | {'yes' if result.separated else 'no'} |",
        f"| Illegal proposals | {result.a_illegal} by `{result.a}`, "
        f"{result.b_illegal} by `{result.b}` |",
        "",
        "## By colour",
        "",
        "A model that only wins with one colour has an opening repertoire, not an edge.",
        "",
        f"| `{result.a}` plays | Games | Score |",
        "|---|---:|---:|",
        f"| white | {len(white)} | {sum(g.score for g in white) / max(len(white), 1):.4f} |",
        f"| black | {len(black)} | {sum(g.score for g in black) / max(len(black), 1):.4f} |",
        "",
        "## Is this enough games?",
        "",
        f"An edge of {abs(result.elo):.0f} Elo takes **{needed}** games to tell from nothing at "
        f"95 %, and this match played **{result.games}**. "
        + (
            "So the number above is a measurement."
            if result.games >= needed
            else "So the number above is a hint, not a measurement: play more games or say so."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def write_match(result: MatchResult, played: Sequence[MatchGame], out_dir: Path | str) -> Path:
    """Write ``report.md`` and ``results.json`` for one match; returns the report."""
    directory = Path(out_dir) / f"{result.a}-vs-{result.b}"
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / "report.md"
    report.write_text(render_markdown(result, played), encoding="utf-8", newline="\n")
    payload = result.model_dump(mode="json")
    payload["played"] = [game.model_dump(mode="json") for game in played]
    (directory / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return report
