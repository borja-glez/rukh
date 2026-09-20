"""Measure one model once per Elo header: does asking it to play weaker actually work?

``GOAL.md`` asks P4 for a monotonic Elo by condition -- 1500 < 2000 < 2400, with intervals -- and
nothing in the harness could answer that, because ``header_elo`` was one number in a config and a
suite run produced one row. This runs the very same suite once per condition, changing that
number and **nothing else**: same opponents, same positions, same puzzles, same seed, same
temperature. Anything that differs between the rows is the header.

Two claims are checked separately and reported separately, because they are not the same claim.
A **monotonic** sweep is one whose point estimates go up with the condition; a **separated** one
is a sweep whose consecutive confidence intervals do not overlap. Point estimates rise by chance
all the time -- that is exactly how D-069 first read a 32-game trial as a win for `<2600>` and
had to take it back at 160 games -- so the sweep prints both and the milestone is judged on the
second.

The cache makes this much cheaper than it sounds: a game is keyed by the weights, the config and
the header, so re-running a sweep after adding one condition only plays the new one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from rukh import paths
from rukh.eval.suite import EvalConfig, run_suite

if TYPE_CHECKING:
    from rukh.eval.suite import SuiteResult

log = logging.getLogger(__name__)

__all__ = ["SweepResult", "SweepRow", "elo_summary", "run_sweep", "sweep_row", "write_sweep"]

WEB_FILE = "artifacts/web/elo-conditioning.json"


class SweepRow(BaseModel):
    """One condition: what the model does when it is told to play at this rating."""

    model_config = ConfigDict(extra="forbid")

    header_elo: int
    elo: float | None = None
    elo_ci: list[float] | None = None
    elo_lower: float | None = None
    elo_upper: float | None = None
    score: float | None = None
    """Points per game over the whole ladder; the raw number the Elo is fitted from."""
    legality: float | None = None
    """Asked to play badly, does it play badly or does it play *illegally*? Not the same thing,
    and the difference is the whole question of whether the header is understood."""
    top1: float | None = None
    puzzles: float | None = None
    first_move_entropy: float | None = None
    stage: str


class SweepResult(BaseModel):
    """The whole sweep and the two verdicts it supports."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    checkpoint: str
    params: int
    date: str
    rows: list[SweepRow]
    monotonic: bool
    """The point estimates rise with the condition. Cheap, and not on its own evidence."""
    separated: bool
    """No two consecutive intervals overlap. This is the one the acceptance bar is read on."""
    span: float | None = None
    """Elo between the weakest and the strongest condition: how much the dial is worth."""

    def table(self) -> str:
        """The sweep as a Markdown table, in the shape the ledger and the lesson quote."""
        lines = [
            "| Asked for | Elo | 95 % CI | legal argmax | top-1 | puzzles | first-move entropy |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
        for row in self.rows:
            ci = _ci_text(row)
            legality = "n/a" if row.legality is None else f"{row.legality * 100:.2f} %"
            top1 = "n/a" if row.top1 is None else f"{row.top1 * 100:.2f} %"
            puzzles = "n/a" if row.puzzles is None else f"{row.puzzles * 100:.2f} %"
            entropy = (
                "n/a" if row.first_move_entropy is None else f"{row.first_move_entropy:.3f} bits"
            )
            elo = "n/a" if row.elo is None else f"{row.elo:.0f}"
            lines.append(
                f"| `<w{row.header_elo:04d}>` | {elo} | {ci} | {legality} | {top1} | "
                f"{puzzles} | {entropy} |"
            )
        return "\n".join(lines)


def _ci_text(row: SweepRow) -> str:
    if row.elo_ci:
        return f"{row.elo_ci[0]:.0f}-{row.elo_ci[1]:.0f}"
    if row.elo_lower is not None:
        return f"> {row.elo_lower:.0f}"
    if row.elo_upper is not None:
        return f"< {row.elo_upper:.0f}"
    return "n/a"


def sweep_row(header_elo: int, result: SuiteResult) -> SweepRow:
    """Pull the columns of one condition out of a full suite result."""
    elo = result.elo
    return SweepRow(
        header_elo=header_elo,
        elo=elo.elo if elo else None,
        elo_ci=[elo.ci_low, elo.ci_high] if elo and not elo.separated else None,
        elo_lower=elo.elo_lower if elo else None,
        elo_upper=elo.elo_upper if elo else None,
        score=elo.score if elo else None,
        legality=result.legality_argmax.rate if result.legality_argmax else None,
        top1=result.accuracy.top1 if result.accuracy else None,
        puzzles=result.puzzles.rate if result.puzzles else None,
        first_move_entropy=(
            result.diversity_detail.first_move_entropy_bits if result.diversity_detail else None
        ),
        stage=result.stage,
    )


def is_monotonic(rows: list[SweepRow]) -> bool:
    """Whether the point estimates rise with the condition, ignoring rows without an Elo."""
    measured = [row.elo for row in rows if row.elo is not None]
    return all(a < b for a, b in zip(measured, measured[1:], strict=False))


def is_separated(rows: list[SweepRow]) -> bool:
    """Whether every consecutive pair of intervals is disjoint, in the right order.

    A one-sided bound (every game won or every game lost) counts as separated only when the
    bound itself proves it, which is the honest reading of "the interval does not overlap".
    """
    usable = [row for row in rows if row.elo is not None]
    if len(usable) < 2:
        return False
    for low, high in zip(usable, usable[1:], strict=False):
        top_of_low = low.elo_ci[1] if low.elo_ci else low.elo_upper
        bottom_of_high = high.elo_ci[0] if high.elo_ci else high.elo_lower
        if top_of_low is None or bottom_of_high is None:
            return False
        if top_of_low >= bottom_of_high:
            return False
    return True


def run_sweep(
    ckpt: Path | str,
    cfg: EvalConfig,
    elos: list[int],
    suite: str = "full",
    use_cache: bool = True,
    device: str | None = None,
    stage: str | None = None,
) -> SweepResult:
    """Run the suite once per condition and collect the rows into one verdict."""
    if len(elos) < 2:
        raise ValueError("a sweep needs at least two conditions to compare")
    if sorted(elos) != elos:
        raise ValueError(f"conditions must be given in increasing order, got {elos}")
    base_stage = stage or cfg.stage or Path(ckpt).parent.name
    rows: list[SweepRow] = []
    last: SuiteResult | None = None
    for header_elo in elos:
        name = f"{base_stage}@{header_elo}"
        log.info("condition %s of %s: %s", len(rows) + 1, len(elos), name)
        per_condition = cfg.model_copy(update={"header_elo": header_elo, "stage": name})
        result, _ = run_suite(ckpt, per_condition, suite=suite, use_cache=use_cache, device=device)
        rows.append(sweep_row(header_elo, result))
        last = result
    assert last is not None
    measured = [row.elo for row in rows if row.elo is not None]
    return SweepResult(
        stage=base_stage,
        checkpoint=Path(ckpt).as_posix(),
        params=last.params,
        date=last.date,
        rows=rows,
        monotonic=is_monotonic(rows),
        separated=is_separated(rows),
        span=(max(measured) - min(measured)) if len(measured) >= 2 else None,
    )


def write_sweep(result: SweepResult, out_dir: Path | str, web: Path | str | None = None) -> Path:
    """Write the sweep's table and JSON, and the file the course reads."""
    directory = Path(out_dir) / f"{result.stage}-elo-sweep"
    directory.mkdir(parents=True, exist_ok=True)
    markdown = directory / "report.md"
    markdown.write_text(_render(result), encoding="utf-8", newline="\n")
    payload = json.loads(result.model_dump_json())
    (directory / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    if web is not None:
        target = paths.resolve(str(web))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return markdown


def _render(result: SweepResult) -> str:
    verdict = "monotonic" if result.monotonic else "**not** monotonic"
    separation = "separated" if result.separated else "**not** separated"
    span = "n/a" if result.span is None else f"{result.span:.0f} Elo"
    return "\n".join(
        [
            f"# Elo by condition of `{result.stage}`",
            "",
            f"- Checkpoint: `{result.checkpoint}`",
            f"- Parameters: {result.params:,}",
            f"- Date: {result.date}",
            "",
            "Every row is the same suite with one number changed: the `<wXXXX> <bXXXX>` header "
            "the model is prompted with. Same opponents, same positions, same puzzles, same "
            "seed, same temperature.",
            "",
            result.table(),
            "",
            "## Verdict",
            "",
            f"- Point estimates: {verdict}.",
            f"- Confidence intervals: {separation}.",
            f"- Span between the weakest and the strongest condition: {span}.",
            "",
            "The two are not the same claim. Point estimates rise by chance often enough that a "
            "monotonic row of numbers is not evidence on its own; the acceptance bar of `GOAL.md` "
            "is read on the intervals.",
            "",
        ]
    )


def elo_summary(rows: list[SweepRow]) -> str:
    """One line per condition for the CLI."""
    lines = [
        f"<w{row.header_elo:04d}>: "
        + ("n/a" if row.elo is None else f"{row.elo:.0f} ({_ci_text(row)})")
        for row in rows
    ]
    return "\n".join(lines)
