"""Elo against Stockfish: play the rungs, then fit the rating that explains the results.

Rungs are Stockfish at ``UCI_Elo`` 1320/1500/1800/2000 (1320 is the engine's floor) plus four
``Skill Level`` rungs below it, so a weak model still has opponents it can score against: with
only losses there is nothing for the fit to latch onto. Both colours are played at every rung.

The fit is the standard logistic (Elo) model with one free parameter,

    P(score) = 1 / (1 + 10 ** ((opponent - elo) / 400)),

maximised by Newton's method: with ``c = ln(10) / 400`` the gradient is ``c * sum(s - p)`` and
the Hessian ``-c**2 * sum(p * (1 - p))``, so each step is ``sum(s - p) / (c * sum(p * (1 - p)))``.
Draws enter as a score of 0.5, which is the usual quasi-likelihood treatment. The 95 % interval
is a percentile bootstrap over the game results, fitted in a single vectorised pass so a
thousand resamples cost milliseconds and no SciPy is needed.

Two honesty notes are built into the numbers rather than left to the reader:

* a game cut short by the context limit is **adjudicated** (shallow engine analysis of the final
  position, or the material count when no engine is around) instead of being booked as a draw,
  because half a point per cut game biases the fit towards the middle of the rungs;
* when every game is a win (or every game is a loss) the likelihood has no maximum inside the
  range of opponents and the bootstrap collapses to a zero-width interval. That is reported as
  ``separated`` with a one-sided likelihood bound instead of a symmetric interval that would be
  a lie.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import chess
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from rukh.config import BaseConfig
from rukh.eval.cache import EvalCache
from rukh.infer import (
    GameResult,
    Player,
    SampleConfig,
    StockfishOpponent,
    adjudicate,
    play_game,
    play_game_with,
)
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

LOG10_400 = math.log(10.0) / 400.0
SUITE = "elo"
MAX_STEP = 400.0
SLACK = 1000.0
"""How far outside the range of opponents the fit is allowed to wander (separation guard)."""
BOUND_SLACK = 4000.0
"""Search range of the one-sided bound when the results are separated."""
MOVE_TIME = 0.1
"""Seconds per move for the engine; below this ``UCI_Elo`` means very little (D-025)."""


class EloRung(BaseConfig):
    """One Stockfish setting and the rating it is assumed to play at."""

    name: str
    elo: int
    uci_elo: int | None = None
    skill: int | None = None

    @model_validator(mode="after")
    def _check(self) -> EloRung:
        if (self.uci_elo is None) == (self.skill is None):
            raise ValueError(f"rung {self.name}: set exactly one of uci_elo and skill")
        return self


# The four ``Skill Level`` rungs carry **measured** ratings, not the nominal 800/950/1100/1250 they
# were given when the harness was written. Those guesses were wrong by 430 to 580 Elo and dragged
# every stage's fit down by about 350 points: the contradiction that exposed them was a model
# scoring 0.725 against a 1320-rated engine while losing 20-0 to one labelled 1250.
#
# The numbers below come from playing the ladder against itself at the suite's own 0.1 s per move,
# 40 games per pair with colours alternated, anchored on ``uci-1320`` (D-070). The method validates
# on its own control: ``uci-1500`` measured +179 Elo over ``uci-1320`` against a nominal +180.
# ``UCI_Elo`` does compress higher up -- ``uci-1800`` measured +215 over ``uci-1500``, not +300 --
# which is a caveat for the top rungs and not for the range these models play in.
# Ordered by measured strength. Note what that ordering shows: **no rung is below 1320**. The
# four ``skill-*`` opponents were put in the ladder to reach under the engine's ``UCI_Elo`` floor
# and they never did, which is why every stage measured lower than it plays.
DEFAULT_RUNGS: list[EloRung] = [
    EloRung(name="uci-1320", elo=1320, uci_elo=1320),
    EloRung(name="skill-0", elo=1381, skill=0),
    EloRung(name="skill-1", elo=1467, skill=1),
    EloRung(name="uci-1500", elo=1500, uci_elo=1500),
    EloRung(name="skill-2", elo=1589, skill=2),
    EloRung(name="skill-3", elo=1678, skill=3),
    EloRung(name="uci-1800", elo=1800, uci_elo=1800),
    EloRung(name="uci-2000", elo=2000, uci_elo=2000),
]


class GameRecord(BaseModel):
    """One played game, from the model's point of view."""

    model_config = ConfigDict(extra="forbid")

    rung: str
    opponent_elo: int
    index: int
    header_elo: int = 1800
    """The Elo the model was asked to play at. Part of the cache key: same rung, different game."""
    model_white: bool
    result: str
    score: float
    plies: int
    illegal_proposals: int
    cut: bool = False
    """The game ran out of context instead of ending on the board."""
    adjudicated: str | None = None
    """How a cut game was decided (``engine depth 8``, ``material count``), or None."""

    def item_id(self) -> str:
        """Cache key. The header is only in it when it is not the default.

        Without the header a run at ``<2600>`` would silently read back the games played at
        ``<1800>``: same rung, same index, completely different game. Leaving the default out
        keeps every game cached before this existed addressable.
        """
        suffix = "" if self.header_elo == 1800 else f":e{self.header_elo}"
        return f"{self.rung}:{self.index}{suffix}"


class RungResult(BaseModel):
    """Aggregated results at one rung."""

    model_config = ConfigDict(extra="forbid")

    name: str
    opponent_elo: int
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    """Average score in [0, 1]."""
    cut: int = 0
    """Games that hit the context limit instead of ending on the board."""
    adjudicated: int = 0
    """Cut games whose result was decided by adjudication."""


class EloResult(BaseModel):
    """The fitted rating with its interval (or one-sided bound), plus the rung breakdown."""

    model_config = ConfigDict(extra="forbid")

    elo: float
    ci_low: float | None = None
    ci_high: float | None = None
    games: int
    score: float
    rungs: list[RungResult]
    cut: int = 0
    adjudicated: int = 0
    separated: bool = False
    """Every game was a win (or every one a loss): the fit is not identified."""
    elo_lower: float | None = None
    """One-sided 95 % lower bound; set when the model won every game."""
    elo_upper: float | None = None
    """One-sided 95 % upper bound; set when the model lost every game."""


def score_of(result: str, model_white: bool) -> float:
    """Score of a finished game from the model's point of view.

    A cut game (``*``) that could not be adjudicated is the only case left at 0.5 by default;
    ``play_rung`` adjudicates cut games before calling this, so that branch is a fallback.
    """
    if result == "1-0":
        return 1.0 if model_white else 0.0
    if result == "0-1":
        return 0.0 if model_white else 1.0
    return 0.5


def _fit_batch(opponents: np.ndarray, scores: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Newton's method on ``(B, n)`` batches of results; returns one rating per row."""
    opponents = np.asarray(opponents, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    low = opponents.min(axis=1) - SLACK
    high = opponents.max(axis=1) + SLACK
    elo = opponents.mean(axis=1)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-LOG10_400 * (elo[:, None] - opponents)))
        gradient = (scores - p).sum(axis=1)
        curvature = (p * (1.0 - p)).sum(axis=1)
        safe = np.maximum(curvature, 1e-12)
        step = np.where(curvature > 1e-12, gradient / (LOG10_400 * safe), 0.0)
        step = np.clip(step, -MAX_STEP, MAX_STEP)
        elo = np.clip(elo + step, low, high)
        if np.max(np.abs(step)) < 1e-6:
            break
    return elo


def fit_elo(opponents: Sequence[float], scores: Sequence[float]) -> float:
    """Maximum-likelihood rating for one set of games."""
    if not len(opponents):
        raise ValueError("cannot fit an Elo without games")
    return float(_fit_batch(np.asarray([opponents]), np.asarray([scores]))[0])


def bootstrap_ci(
    opponents: Sequence[float],
    scores: Sequence[float],
    samples: int = 1_000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval over the games themselves (one fit per resample)."""
    opponents = np.asarray(opponents, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(opponents)
    if n == 0:
        raise ValueError("cannot bootstrap without games")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, size=(samples, n))
    fitted = _fit_batch(opponents[index], scores[index])
    tail = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(fitted, tail)), float(np.percentile(fitted, 100.0 - tail))


def separation(scores: Sequence[float]) -> str | None:
    """``"wins"``, ``"losses"`` or None: whether every game went the same way.

    With no counter-example the likelihood keeps rising as the rating goes to infinity, the fit
    stops at the clamp and every bootstrap resample returns that same clamp, which is where the
    zero-width "95 % CI" came from.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return None
    if np.all(values >= 1.0):
        return "wins"
    if np.all(values <= 0.0):
        return "losses"
    return None


def one_sided_bound(
    opponents: Sequence[float], scores: Sequence[float], level: float = 0.95, lower: bool = True
) -> float:
    """Likelihood bound for separated results: the rating at which the run stops being plausible.

    For an all-win run the bound is the lowest rating under which winning every game still has
    probability ``1 - level``; for an all-loss run it is the highest rating under which losing
    every game does. Solved by bisection on a monotone log-likelihood, so no SciPy is needed.
    """
    rungs = np.asarray(opponents, dtype=np.float64)
    if rungs.size == 0:
        raise ValueError("cannot bound an Elo without games")
    target = math.log(1.0 - level)

    def loglik(elo: float) -> float:
        p = 1.0 / (1.0 + np.exp(-LOG10_400 * (elo - rungs)))
        p = np.clip(p if lower else 1.0 - p, 1e-300, 1.0)
        return float(np.sum(np.log(p)))

    low = float(rungs.min()) - BOUND_SLACK
    high = float(rungs.max()) + BOUND_SLACK
    # ``loglik`` rises with the rating when bounding from below and falls when bounding from
    # above; bisect for the crossing of ``target`` either way.
    for _ in range(200):
        middle = 0.5 * (low + high)
        if (loglik(middle) < target) == lower:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def summarize_rungs(records: Sequence[GameRecord]) -> list[RungResult]:
    """Wins, draws, losses and average score per rung, ordered by opponent rating."""
    by_rung: dict[str, list[GameRecord]] = {}
    for record in records:
        by_rung.setdefault(record.rung, []).append(record)
    results = [
        RungResult(
            name=name,
            opponent_elo=games[0].opponent_elo,
            games=len(games),
            wins=sum(g.score == 1.0 for g in games),
            draws=sum(g.score == 0.5 for g in games),
            losses=sum(g.score == 0.0 for g in games),
            score=sum(g.score for g in games) / len(games),
            cut=sum(g.cut for g in games),
            adjudicated=sum(g.adjudicated is not None for g in games),
        )
        for name, games in by_rung.items()
    ]
    return sorted(results, key=lambda r: (r.opponent_elo, r.name))


def estimate(
    records: Sequence[GameRecord], samples: int = 1_000, seed: int = 0, level: float = 0.95
) -> EloResult:
    """Fit the rating and its interval (or its one-sided bound) from played games."""
    if not records:
        raise ValueError("cannot estimate an Elo without games")
    opponents = [float(record.opponent_elo) for record in records]
    scores = [record.score for record in records]
    elo = fit_elo(opponents, scores)
    rungs = summarize_rungs(records)
    result = EloResult(
        elo=elo,
        games=len(records),
        score=float(np.mean(scores)),
        rungs=rungs,
        cut=sum(rung.cut for rung in rungs),
        adjudicated=sum(rung.adjudicated for rung in rungs),
    )
    apart = separation(scores)
    if apart is None:
        result.ci_low, result.ci_high = bootstrap_ci(
            opponents, scores, samples=samples, seed=seed, level=level
        )
        return result
    result.separated = True
    bound = one_sided_bound(opponents, scores, level=level, lower=apart == "wins")
    if apart == "wins":
        result.elo_lower = bound
    else:
        result.elo_upper = bound
    return result


def record_of(
    outcome: GameResult,
    rung: EloRung,
    index: int,
    model_white: bool,
    engine: chess.engine.SimpleEngine | None = None,
    header_elo: int = 1800,
) -> GameRecord:
    """One played game as a record, adjudicating it first when it was cut short."""
    result = outcome.result
    how: str | None = None
    if outcome.cut:
        result, how = adjudicate(outcome.fen, engine=engine)
    return GameRecord(
        rung=rung.name,
        opponent_elo=rung.elo,
        index=index,
        header_elo=header_elo,
        model_white=model_white,
        result=result,
        score=score_of(result, model_white),
        plies=outcome.plies,
        illegal_proposals=outcome.illegal_proposals,
        cut=outcome.cut,
        adjudicated=how,
    )


def play_rung(
    model: MoveDecoder | None,
    tok: UciTokenizer,
    rung: EloRung,
    games: int,
    cfg: SampleConfig,
    move_time: float = MOVE_TIME,
    max_plies: int | None = None,
    cache: EvalCache | None = None,
    header_elo: int = 1800,
    player: Player | None = None,
) -> list[GameRecord]:
    """Play ``games`` games against one rung, alternating colours, reusing cached games.

    A game that ends with ``*`` (the context ran out) is adjudicated with the rung's own engine
    before it is scored, so it enters the fit as a win, a loss or a draw on the merits of the
    final position.

    ``player`` replaces the decoder, and it is how M4 puts a general language model on this exact
    ladder -- same opponents, same move time, same colours, same adjudication. A comparison whose
    two halves went through different code is not a comparison. When it is given, ``model`` is
    unused and may be ``None``; the seed still varies per game, but a player that cannot be
    seeded simply ignores it.
    """
    records: list[GameRecord] = []
    suffix = "" if header_elo == 1800 else f":e{header_elo}"
    for index in range(games):
        cached = cache.get(SUITE, f"{rung.name}:{index}{suffix}") if cache is not None else None
        if cached is not None:
            records.append(GameRecord.model_validate(cached))
    done = {record.index for record in records}
    missing = [index for index in range(games) if index not in done]
    if not missing:
        return sorted(records, key=lambda r: r.index)

    opponent = StockfishOpponent(elo=rung.uci_elo or 1320, skill=rung.skill, move_time=move_time)
    try:
        for index in missing:
            model_white = index % 2 == 0
            per_game = cfg.model_copy(
                update={"seed": None if cfg.seed is None else cfg.seed + index}
            )
            colour = chess.WHITE if model_white else chess.BLACK
            if player is not None:
                outcome = play_game_with(
                    player,
                    opponent,
                    model_color=colour,
                    white_elo=header_elo,
                    black_elo=header_elo,
                    max_plies=max_plies,
                )
            else:
                assert model is not None, "play_rung needs a model or a player"
                outcome = play_game(
                    model,
                    tok,
                    opponent,
                    per_game,
                    model_color=colour,
                    white_elo=header_elo,
                    black_elo=header_elo,
                    max_plies=max_plies,
                )
            record = record_of(
                outcome, rung, index, model_white, engine=opponent.engine, header_elo=header_elo
            )
            if cache is not None:
                cache.put(SUITE, record.item_id(), record.model_dump())
            records.append(record)
    finally:
        opponent.close()
    return sorted(records, key=lambda r: r.index)


def play_rungs(
    model: MoveDecoder | None,
    tok: UciTokenizer,
    rungs: Sequence[EloRung],
    games: int,
    cfg: SampleConfig,
    move_time: float = MOVE_TIME,
    max_plies: int | None = None,
    cache: EvalCache | None = None,
    header_elo: int = 1800,
    player: Player | None = None,
) -> list[GameRecord]:
    """Play every rung and return all the game records."""
    records: list[GameRecord] = []
    for rung in rungs:
        records.extend(
            play_rung(
                model,
                tok,
                rung,
                games,
                cfg,
                move_time=move_time,
                max_plies=max_plies,
                cache=cache,
                header_elo=header_elo,
                player=player,
            )
        )
    return records
