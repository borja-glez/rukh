"""Reporting: ``report.md`` and ``results.json`` per model, one shared row for the web.

Every stage of the course (base, tiny, small, masters, Elo-conditioned, LoRA, DPO, GRPO, the
external baselines) ends up as one row of the single table of ``docs/spec/02`` §"Componente 5".
``artifacts/web/results.json`` is that table: the course site and the demo read it, so writing a
row is an upsert keyed by ``stage`` rather than an append, and a metric that has not been
measured yet (``delta_cp``, ``diversity``) is written as ``null`` instead of being invented.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rukh.eval.suite import SuiteResult

REPORT_NAME = "report.md"
RESULTS_NAME = "results.json"


class WebRow(BaseModel):
    """One row of the single results table, exactly as the web reads it."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    params: int
    legality: float | None = None
    top1: float | None = None
    top3: float | None = None
    puzzles: dict[str, float] = {}
    elo: float | None = None
    elo_ci: list[float] | None = None
    delta_cp: float | None = None
    diversity: float | None = None
    date: str
    run_id: str | None = None


def row_of(result: SuiteResult) -> WebRow:
    """The table row a finished suite produces."""
    return WebRow(
        stage=result.stage,
        params=result.params,
        legality=result.legality.rate if result.legality else None,
        top1=result.accuracy.top1 if result.accuracy else None,
        top3=result.accuracy.top3 if result.accuracy else None,
        puzzles=result.puzzles.by_band() if result.puzzles else {},
        elo=result.elo.elo if result.elo else None,
        elo_ci=[result.elo.ci_low, result.elo.ci_high] if result.elo else None,
        delta_cp=result.delta_cp,
        diversity=result.diversity,
        date=result.date,
        run_id=result.run_id,
    )


def upsert_row(path: Path, row: WebRow) -> list[dict[str, Any]]:
    """Insert or replace ``row`` in the shared results file and return every row."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            payload = {}
        found = payload.get("rows") if isinstance(payload, dict) else payload
        if isinstance(found, list):
            rows = [item for item in found if isinstance(item, dict)]
    new = row.model_dump()
    rows = [item for item in rows if item.get("stage") != row.stage]
    rows.append(new)
    rows.sort(key=lambda item: str(item.get("stage")))
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"updated_at": datetime.now(UTC).isoformat(timespec="seconds"), "rows": rows}
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return rows


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f} %"


def render_markdown(result: SuiteResult) -> str:
    """The human-readable report: headline numbers first, then every breakdown."""
    legality = result.legality.rate if result.legality else None
    lines: list[str] = [
        f"# Evaluation of `{result.stage}`",
        "",
        f"- Suite: `{result.suite}`",
        f"- Checkpoint: `{result.checkpoint}`",
        f"- Weights SHA-256: `{result.model_sha}`",
        f"- Parameters: {result.params:,}",
        f"- Date: {result.date}",
        f"- MLflow run: {result.run_id or 'not tracked'}",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Legality without the mask | {_percent(legality)} |",
        f"| Top-1 next move | {_percent(result.accuracy.top1 if result.accuracy else None)} |",
        f"| Top-3 next move | {_percent(result.accuracy.top3 if result.accuracy else None)} |",
        f"| Puzzles solved | {_percent(result.puzzles.rate if result.puzzles else None)} |",
    ]
    if result.elo is not None:
        lines.append(
            f"| Estimated Elo | {result.elo.elo:.0f} "
            f"(95 % CI {result.elo.ci_low:.0f}-{result.elo.ci_high:.0f}) |"
        )
    else:
        lines.append("| Estimated Elo | n/a |")
    delta_cp = "n/a" if result.delta_cp is None else f"{result.delta_cp:.1f}"
    diversity = "n/a" if result.diversity is None else f"{result.diversity:.3f}"
    lines.extend(
        [
            f"| Mean centipawn loss | {delta_cp} |",
            f"| Opening diversity | {diversity} |",
            "",
        ]
    )

    if result.legality is not None:
        lines.extend(
            [
                "## Legality",
                "",
                f"{result.legality.legal} of {result.legality.positions} unmasked proposals were "
                "legal moves in validation positions.",
                "",
            ]
        )
    if result.accuracy is not None and result.accuracy.bands:
        lines.extend(
            [
                "## Next-move accuracy by Elo band",
                "",
                "| Band | Positions | Top-1 | Top-3 |",
                "|---|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| {band.band} | {band.positions} | {_percent(band.top1)} | {_percent(band.top3)} |"
            for band in result.accuracy.bands
        )
        lines.append("")
    if result.puzzles is not None and result.puzzles.bands:
        lines.extend(
            [
                "## Puzzles by difficulty band",
                "",
                "| Band | Attempted | Solved | Rate |",
                "|---|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| {band.band} | {band.attempted} | {band.solved} | {_percent(band.rate)} |"
            for band in result.puzzles.bands
        )
        lines.append("")
    if result.elo is not None and result.elo.rungs:
        lines.extend(
            [
                "## Games against Stockfish",
                "",
                "| Rung | Opponent Elo | Games | W | D | L | Score |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| {rung.name} | {rung.opponent_elo} | {rung.games} | {rung.wins} | "
            f"{rung.draws} | {rung.losses} | {rung.score:.3f} |"
            for rung in result.elo.rungs
        )
        lines.append("")
    if result.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in result.notes)
        lines.append("")
    return "\n".join(lines)


class ReportPaths(BaseModel):
    """Where a report landed."""

    model_config = ConfigDict(extra="forbid")

    markdown: str
    results: str
    web: str | None = None


def write_report(
    result: SuiteResult, out_dir: Path, web_results: Path | None = None
) -> ReportPaths:
    """Write ``report.md`` and ``results.json`` and upsert the shared web row."""
    directory = Path(out_dir) / result.stage
    directory.mkdir(parents=True, exist_ok=True)
    markdown = directory / REPORT_NAME
    markdown.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    results = directory / RESULTS_NAME
    results.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    web: str | None = None
    if web_results is not None:
        upsert_row(Path(web_results), row_of(result))
        web = Path(web_results).as_posix()
    return ReportPaths(markdown=markdown.as_posix(), results=results.as_posix(), web=web)
