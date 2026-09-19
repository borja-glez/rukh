"""Tests for rukh.eval.elo: the logistic fit, the bootstrap interval and the rung summary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rukh.eval.cache import EvalCache
from rukh.eval.elo import (
    DEFAULT_RUNGS,
    EloRung,
    GameRecord,
    bootstrap_ci,
    estimate,
    fit_elo,
    one_sided_bound,
    play_rung,
    record_of,
    score_of,
    separation,
    summarize_rungs,
)
from rukh.infer import GameResult, SampleConfig
from rukh.models import DecoderConfig, MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

# The Stockfish game carries only `engine`, so `-m unit` never needs the binary.
unit = pytest.mark.unit

TOY = DecoderConfig(vocab_size=2030, n_layer=2, n_head=2, d_model=32, block=64)

TRUE_ELO = 1500.0
RUNGS = [800.0, 950.0, 1100.0, 1250.0, 1320.0, 1500.0, 1800.0, 2000.0]


def expected(opponents: np.ndarray, elo: float) -> np.ndarray:
    return 1.0 / (1.0 + 10.0 ** ((opponents - elo) / 400.0))


@unit
def test_fit_recovers_the_elo_of_exact_expected_scores() -> None:
    opponents = np.array(RUNGS)
    assert fit_elo(opponents, expected(opponents, TRUE_ELO)) == pytest.approx(TRUE_ELO, abs=0.5)


@unit
def test_fit_recovers_a_known_elo_from_simulated_games() -> None:
    opponents = np.repeat(RUNGS, 125)
    rng = np.random.default_rng(0)
    scores = (rng.random(len(opponents)) < expected(opponents, TRUE_ELO)).astype(float)
    assert fit_elo(opponents, scores) == pytest.approx(TRUE_ELO, abs=30.0)


@unit
def test_draws_count_as_half_a_point() -> None:
    opponents = np.full(200, 1500.0)
    assert fit_elo(opponents, np.full(200, 0.5)) == pytest.approx(1500.0, abs=1.0)


@unit
def test_a_stronger_score_gives_a_higher_elo() -> None:
    opponents = np.full(100, 1600.0)
    weak = fit_elo(opponents, np.full(100, 0.2))
    strong = fit_elo(opponents, np.full(100, 0.8))
    assert weak < 1600.0 < strong


@unit
def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    opponents = np.repeat(RUNGS, 40)
    rng = np.random.default_rng(1)
    scores = (rng.random(len(opponents)) < expected(opponents, TRUE_ELO)).astype(float)
    low, high = bootstrap_ci(opponents, scores, samples=200, seed=0)
    assert low < fit_elo(opponents, scores) < high


@unit
def test_bootstrap_interval_covers_the_truth_in_at_least_ninety_of_a_hundred_runs() -> None:
    """Nominal coverage is 95 %; the seeds are fixed, so the count is deterministic."""
    opponents = np.repeat([1320.0, 1500.0, 1800.0, 2000.0], 40)
    probabilities = expected(opponents, TRUE_ELO)
    rng = np.random.default_rng(0)
    covered = 0
    for simulation in range(100):
        scores = (rng.random(len(opponents)) < probabilities).astype(float)
        low, high = bootstrap_ci(opponents, scores, samples=200, seed=simulation)
        covered += int(low <= TRUE_ELO <= high)
    assert covered >= 90


CUT_WHITE_WINNING = "7k/8/8/8/8/8/8/Q5K1 b - - 0 90"
CUT_LEVEL = "4k3/8/8/8/8/8/8/4K3 w - - 0 90"


def cut_game(fen: str, plies: int = 196) -> GameResult:
    """A game that ran out of context in ``fen``, as ``play_game`` returns it."""
    return GameResult(
        result="*",
        plies=plies,
        illegal_proposals=0,
        moves=[],
        termination="cut_short",
        fen=fen,
    )


@unit
def test_a_cut_game_is_adjudicated_rather_than_scored_as_a_draw() -> None:
    rung = EloRung(name="uci-1500", elo=1500, uci_elo=1500)
    as_white = record_of(cut_game(CUT_WHITE_WINNING), rung, 0, model_white=True)
    assert as_white.cut is True
    assert as_white.adjudicated == "material count"
    assert (as_white.result, as_white.score) == ("1-0", 1.0)

    as_black = record_of(cut_game(CUT_WHITE_WINNING), rung, 1, model_white=False)
    assert (as_black.result, as_black.score) == ("1-0", 0.0)

    level = record_of(cut_game(CUT_LEVEL), rung, 2, model_white=True)
    assert (level.result, level.score) == ("1/2-1/2", 0.5)


@unit
def test_a_finished_game_is_not_adjudicated() -> None:
    rung = EloRung(name="uci-1500", elo=1500, uci_elo=1500)
    finished = GameResult(
        result="0-1",
        plies=40,
        illegal_proposals=2,
        moves=[],
        termination="checkmate",
        fen=CUT_WHITE_WINNING,
    )
    record = record_of(finished, rung, 0, model_white=True)
    assert record.cut is False and record.adjudicated is None
    assert (record.result, record.score) == ("0-1", 0.0)


@unit
def test_the_cut_and_adjudicated_counts_reach_the_summary_and_the_result() -> None:
    rung = EloRung(name="uci-1500", elo=1500, uci_elo=1500)
    games = [
        record_of(cut_game(CUT_WHITE_WINNING), rung, index, index % 2 == 0) for index in range(4)
    ]
    games.extend(records("uci-1320", 1320, [0.0, 1.0]))
    summary = {row.name: row for row in summarize_rungs(games)}
    assert (summary["uci-1500"].cut, summary["uci-1500"].adjudicated) == (4, 4)
    assert (summary["uci-1320"].cut, summary["uci-1320"].adjudicated) == (0, 0)
    fitted = estimate(games, samples=50, seed=0)
    assert (fitted.cut, fitted.adjudicated) == (4, 4)


@unit
def test_separation_is_detected_and_reported_as_a_one_sided_bound() -> None:
    assert separation([1.0, 1.0, 1.0]) == "wins"
    assert separation([0.0, 0.0]) == "losses"
    assert separation([1.0, 0.0]) is None
    assert separation([0.5, 0.5]) is None
    assert separation([]) is None

    all_wins = records("uci-2000", 2000, [1.0] * 40)
    fitted = estimate(all_wins, samples=50, seed=0)
    assert fitted.separated is True
    assert fitted.ci_low is None and fitted.ci_high is None
    assert fitted.elo_lower is not None and fitted.elo_upper is None
    assert fitted.elo_lower > 2000.0

    all_losses = records("skill-0", 800, [0.0] * 40)
    lost = estimate(all_losses, samples=50, seed=0)
    assert lost.separated is True
    assert lost.elo_upper is not None and lost.elo_lower is None
    assert lost.elo_upper < 800.0


@unit
def test_the_one_sided_bound_is_the_rating_that_makes_the_run_implausible() -> None:
    opponents = [1500.0] * 20
    bound = one_sided_bound(opponents, [1.0] * 20, level=0.95, lower=True)
    # Winning 20 games has probability 0.05 exactly at the bound.
    probability = (1.0 / (1.0 + 10.0 ** ((1500.0 - bound) / 400.0))) ** 20
    assert probability == pytest.approx(0.05, abs=1e-3)
    # Ten times as many wins is ten times as much evidence, so the bound rises.
    assert one_sided_bound([1500.0] * 200, [1.0] * 200, lower=True) > bound
    # A single win says almost nothing: the bound sits below the opponent.
    assert one_sided_bound([1500.0], [1.0], lower=True) < 1500.0


@unit
def test_a_mixed_run_still_gets_a_bootstrap_interval() -> None:
    games = records("uci-1320", 1320, [1.0, 0.0] * 20)
    fitted = estimate(games, samples=100, seed=0)
    assert fitted.separated is False
    assert fitted.ci_low is not None and fitted.ci_high is not None
    assert fitted.elo_lower is None and fitted.elo_upper is None


@unit
def test_score_of_reads_the_result_from_the_model_side() -> None:
    assert score_of("1-0", model_white=True) == 1.0
    assert score_of("1-0", model_white=False) == 0.0
    assert score_of("0-1", model_white=False) == 1.0
    assert score_of("1/2-1/2", model_white=True) == 0.5
    assert score_of("*", model_white=True) == 0.5


def records(rung: str, opponent: int, results: list[float]) -> list[GameRecord]:
    return [
        GameRecord(
            rung=rung,
            opponent_elo=opponent,
            index=index,
            model_white=index % 2 == 0,
            result="1-0" if score == 1.0 else "0-1" if score == 0.0 else "1/2-1/2",
            score=score,
            plies=40,
            illegal_proposals=0,
        )
        for index, score in enumerate(results)
    ]


@unit
def test_summarize_rungs_counts_wins_draws_and_losses() -> None:
    summary = summarize_rungs(records("uci-1500", 1500, [1.0, 0.5, 0.0, 1.0]))
    assert len(summary) == 1
    assert (summary[0].wins, summary[0].draws, summary[0].losses) == (2, 1, 1)
    assert summary[0].score == pytest.approx(0.625)


@unit
def test_estimate_reports_rungs_ordered_by_opponent_strength() -> None:
    games = records("uci-1800", 1800, [0.0] * 8) + records("uci-1320", 1320, [1.0] * 8)
    result = estimate(games, samples=100, seed=0)
    assert [rung.name for rung in result.rungs] == ["uci-1320", "uci-1800"]
    assert result.games == 16
    assert result.ci_low <= result.elo <= result.ci_high
    assert 1320.0 < result.elo < 1800.0


@unit
def test_estimate_without_games_is_an_error() -> None:
    with pytest.raises(ValueError, match="without games"):
        estimate([])


@unit
def test_a_rung_needs_exactly_one_of_uci_elo_and_skill() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        EloRung(name="bad", elo=1000)
    with pytest.raises(ValueError, match="exactly one"):
        EloRung(name="bad", elo=1000, uci_elo=1320, skill=3)


@unit
def test_the_default_rungs_are_ordered_and_none_reaches_below_the_engine_floor() -> None:
    """The ladder is sorted by measured strength, and it has no rung under 1320.

    The four ``Skill Level`` rungs were added to reach *below* the engine's ``UCI_Elo`` floor and
    were labelled 800 to 1250 on that assumption. Playing them against ``uci-1320`` showed every
    one of them is stronger than it, by 61 to 358 Elo (D-070), so the assumption was false and
    the labels dragged every stage's fit down about 350 points. This test pins the corrected
    ladder so the old guesses cannot come back unnoticed.
    """
    assert [rung.elo for rung in DEFAULT_RUNGS] == sorted(rung.elo for rung in DEFAULT_RUNGS)
    assert min(rung.elo for rung in DEFAULT_RUNGS) == 1320
    assert {rung.name for rung in DEFAULT_RUNGS if rung.skill is not None} == {
        "skill-0",
        "skill-1",
        "skill-2",
        "skill-3",
    }
    uci = sorted(rung.uci_elo for rung in DEFAULT_RUNGS if rung.uci_elo)
    assert uci == [1320, 1500, 1800, 2000]
    skill = [rung.elo for rung in DEFAULT_RUNGS if rung.skill is not None]
    assert skill == [1381, 1467, 1589, 1678]


@pytest.mark.engine
def test_play_rung_against_stockfish_plays_both_colours_and_caches(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = MoveDecoder(TOY).eval()
    tok = UciTokenizer()
    rung = EloRung(name="skill-0", elo=800, skill=0)
    cfg = SampleConfig(temperature=0.6, top_k=20, mask_illegal=True, seed=0)
    with EvalCache(tmp_path / "cache.sqlite", "sha-a") as cache:
        played = play_rung(model, tok, rung, 2, cfg, move_time=0.01, max_plies=8, cache=cache)
        assert [record.model_white for record in played] == [True, False]
        assert cache.count("elo") == 2
        again = play_rung(model, tok, rung, 2, cfg, move_time=0.01, max_plies=8, cache=cache)
    assert [r.model_dump() for r in again] == [r.model_dump() for r in played]


def test_cache_key_separates_elo_headers() -> None:
    """Two runs of the same rung under different headers are different games.

    Without this the Elo of a model asked to play at 2600 would be read back from the games it
    played at 1800: same rung, same index, silently the wrong answer. The default stays bare so
    every game cached before the header existed is still addressable.
    """
    base = {
        "rung": "uci-1320",
        "opponent_elo": 1320,
        "index": 7,
        "model_white": True,
        "result": "1-0",
        "score": 1.0,
        "plies": 40,
        "illegal_proposals": 0,
    }
    assert GameRecord(**base).item_id() == "uci-1320:7"
    assert GameRecord(**base, header_elo=2600).item_id() == "uci-1320:7:e2600"
    assert GameRecord(**base, header_elo=1800).item_id() == GameRecord(**base).item_id()
