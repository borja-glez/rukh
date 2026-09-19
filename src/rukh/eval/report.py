"""Reporting: ``report.md`` and ``results.json`` per model, one shared row for the web.

Every stage of the course (base, tiny, small, masters, Elo-conditioned, LoRA, DPO, GRPO, the
external baselines) ends up as one row of the single table of ``docs/spec/02`` §"Componente 5".
``artifacts/web/results.json`` is that table: the course site and the demo read it, so writing a
row is an upsert keyed by ``stage`` rather than an append, and a metric that has not been
measured yet (``delta_cp``, ``diversity``) is written as ``null`` instead of being invented.

Two numbers carry their definition with them rather than a footnote somewhere else: legality is
written twice (``argmax``, the headline, and ``sampled``, the demo's own setting), and an Elo
whose games were all wins or all losses is printed as a one-sided bound, in words, instead of a
zero-width "95 % CI".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from rukh.eval.elo import EloResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rukh.eval.encoder import EncoderResult
    from rukh.eval.suite import SuiteResult

REPORT_NAME = "report.md"
RESULTS_NAME = "results.json"


class WebRow(BaseModel):
    """One row of the single results table, exactly as the web reads it."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    params: int
    legality: float | None = None
    """Legality of the most likely token: the headline definition."""
    legality_sampled: float | None = None
    """Legality of a token drawn with the suite's temperature and top-k."""
    top1: float | None = None
    top3: float | None = None
    top1_by_band: dict[str, float] = {}
    top3_by_band: dict[str, float] = {}
    puzzles: dict[str, float] = {}
    elo: float | None = None
    elo_ci: list[float] | None = None
    """95 % bootstrap interval, or ``null`` when the results were separated."""
    elo_lower: float | None = None
    elo_upper: float | None = None
    elo_separated: bool = False
    delta_cp: float | None = None
    diversity: float | None = None
    date: str
    run_id: str | None = None


class EncoderWebRow(BaseModel):
    """One row of the same table for a model that judges positions instead of playing them.

    An encoder has no legality, no Elo and no puzzles, and a decoder has no blunder F1: forcing
    the two into one schema would fill the table with columns that are structurally ``null``.
    The rows share the key (``stage``) and nothing else, and this one says so with ``kind``; a
    row without a ``kind`` is a decoder row, which is what every row written before M3 is.
    """

    model_config = ConfigDict(extra="forbid")

    stage: str
    kind: Literal["encoder"] = "encoder"
    params: int
    blunder_f1: float | None = None
    """F1 of the blunder head on the held-out split."""
    blunder_f1_heuristic: float | None = None
    """F1 of the material baseline on the very same rows."""
    blunder_f1_margin: float | None = None
    """The difference above, in F1 points; ``GOAL.md`` asks for five."""
    blunder_precision: float | None = None
    blunder_recall: float | None = None
    value_pearson: float | None = None
    value_spearman: float | None = None
    result_accuracy: float | None = None
    positions: int = 0
    date: str
    run_id: str | None = None


TableRow = WebRow | EncoderWebRow
"""What ``upsert_row`` accepts: one row of the single results table, of either kind."""


def row_of(result: SuiteResult) -> WebRow:
    """The table row a finished suite produces."""
    elo = result.elo
    interval = (
        [elo.ci_low, elo.ci_high]
        if elo is not None and elo.ci_low is not None and elo.ci_high is not None
        else None
    )
    accuracy = result.accuracy
    return WebRow(
        stage=result.stage,
        params=result.params,
        legality=result.legality_argmax.rate if result.legality_argmax else None,
        legality_sampled=result.legality_sampled.rate if result.legality_sampled else None,
        top1=accuracy.top1 if accuracy else None,
        top3=accuracy.top3 if accuracy else None,
        top1_by_band={band.band: band.top1 for band in accuracy.bands} if accuracy else {},
        top3_by_band={band.band: band.top3 for band in accuracy.bands} if accuracy else {},
        puzzles=result.puzzles.by_band() if result.puzzles else {},
        elo=elo.elo if elo else None,
        elo_ci=interval,
        elo_lower=elo.elo_lower if elo else None,
        elo_upper=elo.elo_upper if elo else None,
        elo_separated=bool(elo.separated) if elo else False,
        delta_cp=result.delta_cp,
        diversity=result.diversity,
        date=result.date,
        run_id=result.run_id,
    )


def encoder_row_of(result: EncoderResult) -> EncoderWebRow:
    """The table row a finished encoder evaluation produces."""
    encoder = result.encoder_blunder
    baseline = result.heuristic_blunder
    value = result.encoder_value
    return EncoderWebRow(
        stage=result.stage,
        params=result.params,
        blunder_f1=encoder.f1 if encoder else None,
        blunder_f1_heuristic=baseline.f1 if baseline else None,
        blunder_f1_margin=result.f1_margin,
        blunder_precision=encoder.precision if encoder else None,
        blunder_recall=encoder.recall if encoder else None,
        value_pearson=value.pearson if value else None,
        value_spearman=value.spearman if value else None,
        result_accuracy=result.result_accuracy,
        positions=result.items,
        date=result.date,
        run_id=result.run_id,
    )


def upsert_row(path: Path, row: TableRow) -> list[dict[str, Any]]:
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


def elo_line(elo: EloResult | None) -> str:
    """The Elo cell: a point estimate with an interval, or with a one-sided bound."""
    if elo is None:
        return "n/a"
    if not elo.separated and elo.ci_low is not None and elo.ci_high is not None:
        return f"{elo.elo:.0f} (95 % CI {elo.ci_low:.0f}-{elo.ci_high:.0f})"
    if elo.elo_lower is not None:
        return f"> {elo.elo_lower:.0f} (one-sided 95 % bound; every game won)"
    if elo.elo_upper is not None:
        return f"< {elo.elo_upper:.0f} (one-sided 95 % bound; every game lost)"
    return f"{elo.elo:.0f} (no interval)"


def render_markdown(result: SuiteResult) -> str:
    """The human-readable report: headline numbers first, then every breakdown."""
    argmax = result.legality_argmax.rate if result.legality_argmax else None
    sampled = result.legality_sampled.rate if result.legality_sampled else None
    sampling = result.legality_sampled
    drawn = (
        ""
        if sampling is None or sampling.temperature is None
        else f" (T={sampling.temperature:g}"
        + ("" if sampling.top_k is None else f", top-k {sampling.top_k}")
        + ")"
    )
    lines: list[str] = [
        f"# Evaluation of `{result.stage}`",
        "",
        f"- Suite: `{result.suite}`",
        f"- Checkpoint: `{result.checkpoint}`",
        f"- Weights SHA-256: `{result.model_sha}`",
        f"- Parameters: {result.params:,}",
        f"- Device: `{result.device}`",
        f"- Date: {result.date}",
        f"- MLflow run: {result.run_id or 'not tracked'}",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Legality without the mask, argmax | {_percent(argmax)} |",
        f"| Legality without the mask, sampled{drawn} | {_percent(sampled)} |",
        f"| Top-1 next move | {_percent(result.accuracy.top1 if result.accuracy else None)} |",
        f"| Top-3 next move | {_percent(result.accuracy.top3 if result.accuracy else None)} |",
        f"| Puzzles solved | {_percent(result.puzzles.rate if result.puzzles else None)} |",
        f"| Estimated Elo | {elo_line(result.elo)} |",
    ]
    delta_cp = "n/a" if result.delta_cp is None else f"{result.delta_cp:.1f}"
    diversity = "n/a" if result.diversity is None else f"{result.diversity:.3f}"
    lines.extend(
        [
            f"| Mean centipawn loss | {delta_cp} |",
            f"| Opening diversity | {diversity} |",
            "",
        ]
    )

    if result.legality_argmax is not None or result.legality_sampled is not None:
        lines.extend(
            [
                "## Legality",
                "",
                "Two rates, because they answer different questions. **argmax** is the share of "
                "validation positions whose single most likely token is a legal move, with no "
                "temperature, no top-k and no mask: it is a property of the weights and it is "
                "the definition behind the \u2265 99 % bar of `GOAL.md`. **sampled** draws the "
                "token exactly as the demo does" + drawn.strip() + ", so it is what a player "
                "would meet with the mask switched off, and it is always the lower of the two.",
                "",
                "| Definition | Positions | Legal | Rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for measured in (result.legality_argmax, result.legality_sampled):
            if measured is not None:
                lines.append(
                    f"| {measured.mode} | {measured.positions} | {measured.legal} | "
                    f"{_percent(measured.rate)} |"
                )
        lines.append("")
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
                f"Prompt: {result.puzzles.prompt_style}.",
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
    if result.elo is not None:
        lines.extend(["## Games against Stockfish", ""])
        if result.elo.rungs:
            lines.extend(
                [
                    "| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            lines.extend(
                f"| {rung.name} | {rung.opponent_elo} | {rung.games} | {rung.wins} | "
                f"{rung.draws} | {rung.losses} | {rung.score:.3f} | {rung.cut} | "
                f"{rung.adjudicated} |"
                for rung in result.elo.rungs
            )
            lines.append("")
        lines.extend(
            [
                f"{result.elo.cut} of {result.elo.games} games hit the context limit; "
                f"{result.elo.adjudicated} of those were adjudicated on the final position "
                "(shallow engine analysis, or the material count when no engine was available) "
                "rather than scored as draws.",
                "",
            ]
        )
        if result.elo.separated:
            lines.extend(
                [
                    "Every game went the same way, so the logistic fit has no maximum inside the "
                    "range of opponents and a bootstrap interval would be zero wide. The table "
                    "gives a one-sided 95 % likelihood bound instead: the rating is on that side "
                    "of the bound, and the games say nothing about how far.",
                    "",
                ]
            )
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


def write_report_files(
    stage: str,
    markdown_text: str,
    payload: Any,
    out_dir: Path,
    web_results: Path | None,
    row: TableRow,
) -> ReportPaths:
    """Write one stage's ``report.md`` and ``results.json`` and upsert its row for the web.

    The two suites (the decoder's and the encoder's) measure different things and render
    different reports, but they write them in the same place, in the same shape and into the
    same table, so the part that is the same lives here once.
    """
    directory = Path(out_dir) / stage
    directory.mkdir(parents=True, exist_ok=True)
    markdown = directory / REPORT_NAME
    markdown.write_text(markdown_text, encoding="utf-8", newline="\n")
    results = directory / RESULTS_NAME
    results.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    web: str | None = None
    if web_results is not None:
        upsert_row(Path(web_results), row)
        web = Path(web_results).as_posix()
    return ReportPaths(markdown=markdown.as_posix(), results=results.as_posix(), web=web)


def write_report(
    result: SuiteResult, out_dir: Path, web_results: Path | None = None
) -> ReportPaths:
    """Write ``report.md`` and ``results.json`` and upsert the shared web row."""
    return write_report_files(
        result.stage,
        render_markdown(result),
        json.loads(result.model_dump_json()),
        Path(out_dir),
        web_results,
        row_of(result),
    )
