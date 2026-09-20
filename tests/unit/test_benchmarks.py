"""`docs/benchmarks.md` is rendered from the measured rows, never typed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rukh.eval.report import BENCHMARKS_MARK, render_benchmarks, write_benchmarks

pytestmark = pytest.mark.unit

ROWS = [
    {
        "stage": "medium-elo",
        "kind": None,
        "legality": 0.998,
        "top1": 0.541,
        "top3": 0.83,
        "puzzles": {"1000-1500": 0.57, "1500-2000": 0.37, "2000+": 0.17},
        "elo": 1497.9,
        "elo_ci": [1435.2, 1552.3],
        "delta_cp": None,
        "first_move_entropy": 1.7408,
        "date": "2026-09-20",
    },
    {"stage": "encoder", "kind": "encoder", "blunder_f1": 0.18, "date": "2026-09-19"},
    {
        "stage": "all-wins",
        "elo": None,
        "elo_separated": True,
        "elo_lower": 2000.0,
        "date": "2026-09-20",
    },
]


def test_the_encoder_rows_stay_out():
    """They do not share one column with a model that plays, so they have their own table."""
    table = render_benchmarks(ROWS)
    assert "`medium-elo`" in table
    assert "`encoder`" not in table
    assert "2 etapas medidas" in table


def test_a_measurement_that_is_not_there_says_so_instead_of_a_zero():
    table = render_benchmarks(ROWS)
    row = next(line for line in table.splitlines() if "`medium-elo`" in line)
    # `delta_cp` has never been measured in this project; an invented 0.0 would read as perfect.
    assert "| n/a |" in row
    assert "1498 (1435-1552)" in row
    assert "1.7408" in row


def test_a_one_sided_bound_is_printed_as_a_bound():
    """Every game won has no interval, and a zero-width one would be a lie about precision."""
    row = next(line for line in render_benchmarks(ROWS).splitlines() if "`all-wins`" in line)
    assert "> 2000" in row


def test_only_the_results_section_is_rewritten(tmp_path: Path):
    document = tmp_path / "benchmarks.md"
    document.write_text(
        f"# Título\n\nCómo se mide cada columna.\n\n{BENCHMARKS_MARK}\n\n| viejo |\n|---|\n",
        encoding="utf-8",
    )
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"rows": ROWS}), encoding="utf-8")

    assert write_benchmarks(results, document) == 2
    written = document.read_text(encoding="utf-8")
    assert written.startswith("# Título\n\nCómo se mide cada columna.\n")
    assert "| viejo |" not in written
    assert "`medium-elo`" in written


def test_a_document_without_the_heading_is_refused(tmp_path: Path):
    document = tmp_path / "benchmarks.md"
    document.write_text("# Sin tabla\n", encoding="utf-8")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"rows": ROWS}), encoding="utf-8")
    with pytest.raises(ValueError, match="Resultados"):
        write_benchmarks(results, document)
