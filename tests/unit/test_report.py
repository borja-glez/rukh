"""Tests for rukh.eval.report: the Markdown report, results.json and the shared web table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rukh.eval.accuracy import AccuracyResult, BandAccuracy
from rukh.eval.elo import EloResult, RungResult
from rukh.eval.legality import LegalityResult
from rukh.eval.puzzles import BandPuzzles, PuzzleResult
from rukh.eval.report import (
    RESULTS_NAME,
    WebRow,
    elo_line,
    render_markdown,
    row_of,
    upsert_row,
    write_report,
)
from rukh.eval.suite import SuiteResult

pytestmark = pytest.mark.unit


def result(stage: str = "tiny", elo: float = 1234.0) -> SuiteResult:
    return SuiteResult(
        stage=stage,
        suite="quick",
        checkpoint="checkpoints/tiny/best.pt",
        model_sha="abc123",
        params=5_000_000,
        date="2026-09-19",
        device="cuda",
        run_id="run-1",
        legality_argmax=LegalityResult(positions=1000, legal=987, rate=0.987, mode="argmax"),
        legality_sampled=LegalityResult(
            positions=1000, legal=901, rate=0.901, mode="sampled", temperature=0.6, top_k=20
        ),
        accuracy=AccuracyResult(
            positions=1000,
            top1=0.31,
            top3=0.52,
            bands=[BandAccuracy(band="1800-2000", positions=1000, top1=0.31, top3=0.52)],
        ),
        puzzles=PuzzleResult(
            attempted=400,
            solved=40,
            rate=0.1,
            bands=[
                BandPuzzles(band="1000-1500", attempted=200, solved=30, rate=0.15),
                BandPuzzles(band="2000+", attempted=200, solved=10, rate=0.05),
            ],
        ),
        elo=EloResult(
            elo=elo,
            ci_low=elo - 60,
            ci_high=elo + 60,
            games=160,
            score=0.42,
            cut=4,
            adjudicated=4,
            rungs=[
                RungResult(
                    name="uci-1320",
                    opponent_elo=1320,
                    games=80,
                    wins=40,
                    draws=5,
                    losses=35,
                    score=0.53,
                    cut=4,
                    adjudicated=4,
                )
            ],
        ),
        notes=["Elo skipped for the 2000 rung"],
    )


def test_the_markdown_report_carries_every_headline_number() -> None:
    text = render_markdown(result())
    assert "# Evaluation of `tiny`" in text
    assert "98.7 %" in text  # legality, argmax
    assert "90.1 %" in text  # legality, sampled
    assert "31.0 %" in text  # top-1
    assert "1234 (95 % CI 1174-1294)" in text
    assert "| 1000-1500 | 200 | 30 | 15.0 %" in text
    assert "Elo skipped for the 2000 rung" in text


def test_the_report_spells_out_both_legality_definitions() -> None:
    text = render_markdown(result())
    assert "Legality without the mask, argmax" in text
    assert "Legality without the mask, sampled (T=0.6, top-k 20)" in text
    assert "| argmax | 1000 | 987 | 98.7 % |" in text
    assert "| sampled | 1000 | 901 | 90.1 % |" in text


def test_the_report_names_the_device_and_the_adjudicated_games() -> None:
    text = render_markdown(result())
    assert "- Device: `cuda`" in text
    assert "were adjudicated on the final position" in text
    assert "| uci-1320 | 1320 | 80 | 40 | 5 | 35 | 0.530 | 4 | 4 |" in text


def test_a_separated_fit_is_reported_as_a_one_sided_bound() -> None:
    separated = EloResult(
        elo=3000.0,
        games=80,
        score=1.0,
        rungs=[],
        separated=True,
        elo_lower=2450.0,
    )
    assert elo_line(separated) == "> 2450 (one-sided 95 % bound; every game won)"
    text = render_markdown(result().model_copy(update={"elo": separated}))
    assert "> 2450 (one-sided 95 % bound; every game won)" in text
    assert "a bootstrap interval would be zero wide" in text
    assert row_of(result().model_copy(update={"elo": separated})).elo_ci is None


def test_missing_metrics_are_reported_as_not_available() -> None:
    bare = result().model_copy(update={"elo": None, "puzzles": None, "delta_cp": None})
    text = render_markdown(bare)
    assert "| Estimated Elo | n/a |" in text
    assert "| Mean centipawn loss | n/a |" in text


def test_write_report_writes_results_that_load_back(tmp_path: Path) -> None:
    paths = write_report(result(), tmp_path / "eval", tmp_path / "web" / "results.json")
    results = Path(paths.results)
    assert results.name == RESULTS_NAME
    assert results.parent.name == "tiny"
    reloaded = SuiteResult.model_validate_json(results.read_text(encoding="utf-8"))
    assert reloaded.model_dump() == result().model_dump()
    assert "# Evaluation of `tiny`" in Path(paths.markdown).read_text(encoding="utf-8")


def test_the_web_row_holds_the_columns_of_the_single_table() -> None:
    row = row_of(result())
    assert row.model_dump() == {
        "stage": "tiny",
        "params": 5_000_000,
        "legality": 0.987,
        "legality_sampled": 0.901,
        "top1": 0.31,
        "top3": 0.52,
        "top1_by_band": {"1800-2000": 0.31},
        "top3_by_band": {"1800-2000": 0.52},
        "puzzles": {"1000-1500": 0.15, "2000+": 0.05},
        "elo": 1234.0,
        "elo_ci": [1174.0, 1294.0],
        "elo_lower": None,
        "elo_upper": None,
        "elo_separated": False,
        "delta_cp": None,
        "diversity": None,
        "date": "2026-09-19",
        "run_id": "run-1",
    }


def test_upserting_the_same_stage_twice_keeps_one_updated_row(tmp_path: Path) -> None:
    path = tmp_path / "web" / "results.json"
    upsert_row(path, row_of(result(elo=1100.0)))
    rows = upsert_row(path, row_of(result(elo=1300.0)))
    assert len(rows) == 1
    assert rows[0]["elo"] == 1300.0
    document = json.loads(path.read_text(encoding="utf-8"))
    assert [row["elo"] for row in document["rows"]] == [1300.0]
    assert "updated_at" in document


def test_a_second_stage_is_appended_and_sorted(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    upsert_row(path, row_of(result(stage="small")))
    rows = upsert_row(path, row_of(result(stage="base")))
    assert [row["stage"] for row in rows] == ["base", "small"]


def test_a_corrupt_table_is_replaced_rather_than_crashing(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text("not json", encoding="utf-8")
    rows = upsert_row(path, WebRow(stage="tiny", params=1, date="2026-09-19"))
    assert [row["stage"] for row in rows] == ["tiny"]
