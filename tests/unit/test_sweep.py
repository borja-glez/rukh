"""Tests for rukh.eval.sweep: the two verdicts the acceptance bar of P4 is read on.

The bar is "Elo by condition monotonic, with intervals". Those are two different claims and the
code has to keep them apart: point estimates rise by chance often enough that a monotonic row of
numbers proves nothing on its own -- D-069 read a 32-game trial as a win for `<2600>` and had to
take it back at 160 games.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rukh.eval.sweep import (
    SweepResult,
    SweepRow,
    elo_summary,
    is_monotonic,
    is_separated,
    run_sweep,
    write_sweep,
)

pytestmark = pytest.mark.unit


def _row(header_elo: int, elo: float | None, ci: tuple[float, float] | None, **extra: object):
    return SweepRow(
        header_elo=header_elo,
        elo=elo,
        elo_ci=list(ci) if ci else None,
        stage=f"s@{header_elo}",
        **extra,  # type: ignore[arg-type]
    )


def _result(rows: list[SweepRow]) -> SweepResult:
    measured = [row.elo for row in rows if row.elo is not None]
    return SweepResult(
        stage="toy",
        checkpoint="checkpoints/toy/best.pt",
        params=1234,
        date="2026-09-20",
        rows=rows,
        monotonic=is_monotonic(rows),
        separated=is_separated(rows),
        span=(max(measured) - min(measured)) if len(measured) >= 2 else None,
    )


def test_rising_point_estimates_are_monotonic() -> None:
    rows = [_row(1500, 1100, (900, 1300)), _row(2000, 1200, (1000, 1400))]
    assert is_monotonic(rows)


def test_a_dip_anywhere_breaks_monotonicity() -> None:
    rows = [_row(1500, 1100, None), _row(2000, 1300, None), _row(2400, 1250, None)]
    assert not is_monotonic(rows)


def test_overlapping_intervals_are_not_separated() -> None:
    """The case D-069 ran into: the numbers rise and the measurement does not support it."""
    rows = [_row(1500, 1100, (900, 1300)), _row(2000, 1200, (1000, 1400))]
    assert is_monotonic(rows)
    assert not is_separated(rows)


def test_disjoint_intervals_are_separated() -> None:
    rows = [_row(1500, 1000, (950, 1050)), _row(2000, 1300, (1250, 1350))]
    assert is_separated(rows)


def test_touching_intervals_are_not_separated() -> None:
    rows = [_row(1500, 1000, (950, 1200)), _row(2000, 1300, (1200, 1350))]
    assert not is_separated(rows)


def test_a_condition_without_an_elo_cannot_be_separated() -> None:
    rows = [_row(1500, None, None), _row(2000, 1300, (1250, 1350))]
    assert not is_separated(rows)


def test_a_one_sided_bound_still_counts_when_the_bound_proves_it() -> None:
    """Every game lost gives an upper bound; it separates if the next row starts above it."""
    rows = [
        SweepRow(header_elo=1500, elo=700, elo_upper=800, stage="a"),
        SweepRow(header_elo=2000, elo=1300, elo_ci=[1250, 1350], stage="b"),
    ]
    assert is_separated(rows)


def test_a_single_condition_is_never_a_sweep() -> None:
    assert not is_separated([_row(1800, 1300, (1250, 1350))])


def test_the_table_names_the_token_the_model_was_actually_shown() -> None:
    """`<w1500>`, not "1500": the header is a token and the reader has to be able to find it."""
    result = _result(
        [
            _row(1500, 1000.4, (950, 1050), legality=0.987, top1=0.51, puzzles=0.22),
            _row(2400, 1400.6, (1350, 1450), legality=0.998, top1=0.54, puzzles=0.38),
        ]
    )
    table = result.table()
    assert "| `<w1500>` | 1000 | 950-1050 | 98.70 % | 51.00 % | 22.00 % | n/a |" in table
    assert "`<w2400>`" in table


def test_the_report_says_both_verdicts_and_the_span(tmp_path: Path) -> None:
    result = _result([_row(1500, 1000, (950, 1050)), _row(2400, 1400, (1350, 1450))])
    markdown = write_sweep(result, tmp_path)
    text = markdown.read_text(encoding="utf-8")
    assert "- Point estimates: monotonic." in text
    assert "- Confidence intervals: separated." in text
    assert "Span between the weakest and the strongest condition: 400 Elo." in text
    payload = json.loads((markdown.parent / "results.json").read_text(encoding="utf-8"))
    assert payload["monotonic"] is True and payload["separated"] is True


def test_a_failed_sweep_says_so_in_words_rather_than_hiding_it(tmp_path: Path) -> None:
    result = _result([_row(1500, 1300, (1200, 1400)), _row(2400, 1250, (1150, 1350))])
    text = write_sweep(result, tmp_path).read_text(encoding="utf-8")
    assert "- Point estimates: **not** monotonic." in text
    assert "- Confidence intervals: **not** separated." in text


def test_the_web_file_is_written_where_the_course_reads_it(tmp_path: Path) -> None:
    result = _result([_row(1500, 1000, (950, 1050)), _row(2400, 1400, (1350, 1450))])
    web = tmp_path / "web" / "elo-conditioning.json"
    write_sweep(result, tmp_path, web)
    payload = json.loads(web.read_text(encoding="utf-8"))
    assert [row["header_elo"] for row in payload["rows"]] == [1500, 2400]


def test_the_summary_line_is_one_condition_per_row() -> None:
    rows = [_row(1500, 1000, (950, 1050)), _row(2400, None, None)]
    assert elo_summary(rows).splitlines() == ["<w1500>: 1000 (950-1050)", "<w2400>: n/a"]


def test_conditions_out_of_order_are_refused(tmp_path: Path) -> None:
    from rukh.eval.suite import EvalConfig

    with pytest.raises(ValueError, match="increasing order"):
        run_sweep(tmp_path / "x.pt", EvalConfig(), [2400, 1500])


def test_one_condition_is_refused(tmp_path: Path) -> None:
    from rukh.eval.suite import EvalConfig

    with pytest.raises(ValueError, match="at least two"):
        run_sweep(tmp_path / "x.pt", EvalConfig(), [1800])
