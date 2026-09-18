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
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import chess
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from rukh.config import BaseConfig
from rukh.eval.cache import EvalCache
from rukh.infer import SampleConfig, StockfishOpponent, play_game
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

LOG10_400 = math.log(10.0) / 400.0
SUITE = "elo"
MAX_STEP = 400.0
SLACK = 1000.0
"""How far outside the range of opponents the fit is allowed to wander (separation guard)."""


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


# ``Skill Level`` ratings are nominal anchors for the rungs below the engine's 1320 floor: they
# are not measured strengths, so a model whose fit leans on them is reported with that caveat.
DEFAULT_RUNGS: list[EloRung] = [
    EloRung(name="skill-0", elo=800, skill=0),
    EloRung(name="skill-1", elo=950, skill=1),
    EloRung(name="skill-2", elo=1100, skill=2),
    EloRung(name="skill-3", elo=1250, skill=3),
    EloRung(name="uci-1320", elo=1320, uci_elo=1320),
    EloRung(name="uci-1500", elo=1500, uci_elo=1500),
    EloRung(name="uci-1800", elo=1800, uci_elo=1800),
    EloRung(name="uci-2000", elo=2000, uci_elo=2000),
]


class GameRecord(BaseModel):
    """One played game, from the model's point of view."""

    model_config = ConfigDict(extra="forbid")

    rung: str
    opponent_elo: int
    index: int
    model_white: bool
    result: str
    score: float
    plies: int
    illegal_proposals: int

    def item_id(self) -> str:
        return f"{self.rung}:{self.index}"


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


class EloResult(BaseModel):
    """The fitted rating with its bootstrap interval, plus the rung breakdown."""

    model_config = ConfigDict(extra="forbid")

    elo: float
    ci_low: float
    ci_high: float
    games: int
    score: float
    rungs: list[RungResult]


def score_of(result: str, model_white: bool) -> float:
    """Score of a finished game from the model's point of view (0.5 for a draw or a cut game)."""
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
        )
        for name, games in by_rung.items()
    ]
    return sorted(results, key=lambda r: (r.opponent_elo, r.name))


def estimate(
    records: Sequence[GameRecord], samples: int = 1_000, seed: int = 0, level: float = 0.95
) -> EloResult:
    """Fit the rating and its interval from played games."""
    if not records:
        raise ValueError("cannot estimate an Elo without games")
    opponents = [float(record.opponent_elo) for record in records]
    scores = [record.score for record in records]
    elo = fit_elo(opponents, scores)
    low, high = bootstrap_ci(opponents, scores, samples=samples, seed=seed, level=level)
    return EloResult(
        elo=elo,
        ci_low=low,
        ci_high=high,
        games=len(records),
        score=float(np.mean(scores)),
        rungs=summarize_rungs(records),
    )


def play_rung(
    model: MoveDecoder,
    tok: UciTokenizer,
    rung: EloRung,
    games: int,
    cfg: SampleConfig,
    move_time: float = 0.05,
    max_plies: int | None = None,
    cache: EvalCache | None = None,
) -> list[GameRecord]:
    """Play ``games`` games against one rung, alternating colours, reusing cached games."""
    records: list[GameRecord] = []
    for index in range(games):
        cached = cache.get(SUITE, f"{rung.name}:{index}") if cache is not None else None
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
            outcome = play_game(
                model,
                tok,
                opponent,
                cfg.model_copy(update={"seed": None if cfg.seed is None else cfg.seed + index}),
                model_color=chess.WHITE if model_white else chess.BLACK,
                max_plies=max_plies,
            )
            record = GameRecord(
                rung=rung.name,
                opponent_elo=rung.elo,
                index=index,
                model_white=model_white,
                result=outcome.result,
                score=score_of(outcome.result, model_white),
                plies=outcome.plies,
                illegal_proposals=outcome.illegal_proposals,
            )
            if cache is not None:
                cache.put(SUITE, record.item_id(), record.model_dump())
            records.append(record)
    finally:
        opponent.close()
    return sorted(records, key=lambda r: r.index)


def play_rungs(
    model: MoveDecoder,
    tok: UciTokenizer,
    rungs: Sequence[EloRung],
    games: int,
    cfg: SampleConfig,
    move_time: float = 0.05,
    max_plies: int | None = None,
    cache: EvalCache | None = None,
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
            )
        )
    return records
