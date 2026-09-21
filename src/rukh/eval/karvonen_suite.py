"""The decoder's suite on the public baseline, so the table has one row nobody here trained.

Everything is borrowed, on purpose: the validation positions of ``rukh.eval.legality``, the
puzzle split of ``rukh.eval.puzzles``, the Stockfish ladder (node budget included) and the
bootstrap of ``rukh.eval.elo``, and the failure accounting of ``rukh.eval.qwen_source``. Only
the player differs. A baseline measured by different code from the models it is compared with
would be a second harness, and the point of the row is to check the first one.

Two things this model cannot be given, and the report says so rather than papering over them.
It has no rating header, so ``header_elo`` is silently not a condition for this row. And it
cannot read a FEN, so a puzzle whose parquet carries no game prefix is answered blind -- from a
transcript that starts at move 23 with no moves in it -- and the number of such prompts is
written next to the puzzle rate.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chess
from pydantic import BaseModel, ConfigDict

from rukh import paths
from rukh.eval.elo import EloResult, estimate, play_rungs
from rukh.eval.karvonen import (
    CHECKPOINT_FILE,
    HF_REPO,
    KarvonenPlayer,
    ensure_karvonen,
    karvonen_source,
    load_karvonen,
)
from rukh.eval.puzzles import PuzzleResult, load_puzzles, run_puzzles
from rukh.eval.qwen_source import QwenStats
from rukh.eval.report import WebRow, elo_line, write_report_files
from rukh.eval.suite import EvalConfig, _positions
from rukh.tokenize.uci_vocab import UciTokenizer

if TYPE_CHECKING:
    from rukh.eval.legality import Position
    from rukh.eval.report import ReportPaths

log = logging.getLogger(__name__)

__all__ = ["KarvonenResult", "evaluate_karvonen", "render_markdown", "row_of", "run_karvonen_suite"]

DEFAULT_STAGE = "karvonen-8l"


class KarvonenResult(BaseModel):
    """One evaluation of the public baseline on the decoder's own suite."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    source: str
    """Where the weights come from: the Hub repository and file of their author."""
    checkpoint: str
    model_sha: str
    params: int
    date: str
    device: str
    positions: int
    written: QwenStats
    """How its answers came out, by failure mode, over the validation positions only."""
    top1: float | None = None
    puzzles: PuzzleResult | None = None
    puzzles_blind: int = 0
    """Puzzle prompts that carried no game prefix and were answered from the FEN position alone,
    which this model cannot read. A rate over many of these is a floor, not a measurement."""
    elo: EloResult | None = None
    notes: list[str] = []
    config: dict[str, Any] = {}


def _board_of(position: Position) -> chess.Board:
    board = chess.Board()
    for uci in position.moves:
        board.push(chess.Move.from_uci(uci))
    return board


def evaluate_karvonen(
    cfg: EvalConfig,
    stage: str = DEFAULT_STAGE,
    use_cache: bool = True,
    device: str | None = None,
    checkpoint: Path | str | None = None,
) -> KarvonenResult:
    """Run legality, next-move accuracy, puzzles and Elo on the published nanoGPT."""
    from rukh.engine import EngineNotFound
    from rukh.eval.cache import EvalCache, config_sha, file_sha
    from rukh.train import pick_device

    ckpt = Path(checkpoint) if checkpoint is not None else ensure_karvonen()
    where = device or pick_device()
    model = load_karvonen(ckpt, device=where)
    sha = file_sha(ckpt)
    tok = UciTokenizer()
    notes: list[str] = [
        "this model has no rating header: header_elo does not condition this row",
    ]
    positions: list[Position] = _positions(cfg, tok, notes)
    player = KarvonenPlayer(model, temperature=cfg.temperature, top_k=cfg.top_k, seed=cfg.seed)

    log.info("legality and accuracy over %s positions", min(len(positions), cfg.accuracy_positions))
    scored = positions[: cfg.accuracy_positions]
    hits = 0
    for position in scored:
        move, _ = player.propose(_board_of(position))
        if move is not None and position.target is not None and move.uci() == position.target:
            hits += 1
    written = player.stats.model_copy(deep=True)

    result = KarvonenResult(
        stage=stage,
        source=f"{HF_REPO}/{CHECKPOINT_FILE}",
        checkpoint=ckpt.as_posix(),
        model_sha=sha,
        params=sum(p.numel() for p in model.parameters()),
        date=datetime.now(UTC).date().isoformat(),
        device=str(next(model.parameters()).device),
        positions=len(scored),
        written=written,
        top1=hits / len(scored) if scored else None,
        config=cfg.model_dump(mode="json"),
    )

    cache_path = paths.resolve(cfg.cache_db) if use_cache else None
    games_cache = EvalCache(
        cache_path, sha, enabled=use_cache, config_sha=config_sha(cfg.cache_fields("games"))
    )
    puzzles_cache = EvalCache(
        cache_path, sha, enabled=use_cache, config_sha=config_sha(cfg.cache_fields("puzzles"))
    )
    try:
        puzzle_path = paths.resolve(cfg.puzzles)
        if puzzle_path.is_file():
            items = load_puzzles(
                puzzle_path, cfg.puzzles_per_band, seed=cfg.seed, split=cfg.puzzle_split
            )
            log.info("puzzles: %s", len(items))
            source = karvonen_source(player, tok)
            result.puzzles = run_puzzles(
                source,
                tok,
                items,
                cache=puzzles_cache,
                header_elo=cfg.header_elo,
                force_header=cfg.force_header,
            )
            result.puzzles_blind = source.blind
            if source.blind:
                notes.append(
                    f"{source.blind} puzzle prompts had no game prefix and were answered from "
                    "the position alone, which this model cannot read: the rate is a floor"
                )
        else:
            notes.append(f"puzzles not found at {puzzle_path}: puzzle suite skipped")
        if cfg.elo_games:
            log.info("elo: %s games per rung", cfg.elo_games)
            try:
                records = play_rungs(
                    None,
                    tok,
                    cfg.elo_rungs,
                    cfg.elo_games,
                    cfg.sampling(),
                    move_time=cfg.elo_move_time,
                    max_plies=cfg.elo_max_plies,
                    cache=games_cache,
                    header_elo=cfg.header_elo,
                    player=player,
                    nodes=cfg.elo_nodes,
                )
            except EngineNotFound as exc:
                notes.append(f"Elo skipped: {exc}")
            else:
                result.elo = estimate(records, samples=cfg.bootstrap, seed=cfg.seed)
    finally:
        games_cache.close()
        puzzles_cache.close()
    # ``written`` stays the snapshot over the validation positions, as in the Qwen row: the
    # player keeps counting through puzzles and games, and folding those in would put a
    # different denominator under the one number that faces the decoder's legality directly.
    result.notes = notes
    return result


def render_markdown(result: KarvonenResult) -> str:
    """The report, with the failure breakdown the decoder's report has no column for."""
    stats = result.written

    def share(count: int) -> str:
        return f"{100 * count / stats.asked:.2f} %" if stats.asked else "n/a"

    lines = [
        f"# Evaluation of `{result.stage}` (public baseline)",
        "",
        f"- Source: `{result.source}`",
        f"- Checkpoint: `{result.checkpoint}`",
        f"- Weights SHA-256: `{result.model_sha}`",
        f"- Parameters: {result.params:,}",
        f"- Device: `{result.device}`",
        f"- Date: {result.date}",
        "",
        "A character-level nanoGPT trained by Adam Karvonen on 16 M Lichess games written as "
        "`;1.e4 e5 2.Nf3`, measured here with the same positions, puzzles and Stockfish ladder "
        "as every other row. It was not trained by this course and nothing about it was tuned.",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Legal moves written, no mask | {share(stats.legal)} |",
        f"| Top-1 next move | {'n/a' if result.top1 is None else f'{result.top1 * 100:.1f} %'} |",
        f"| Puzzles solved | "
        f"{'n/a' if result.puzzles is None else f'{result.puzzles.rate * 100:.1f} %'} |",
        f"| Estimated Elo | {elo_line(result.elo)} |",
        "",
        "## How the answers failed",
        "",
        "| Outcome | Count | Share |",
        "|---|---:|---:|",
        f"| legal | {stats.legal} | {share(stats.legal)} |",
        f"| illegal here | {stats.illegal} | {share(stats.illegal)} |",
        f"| not a move at all | {stats.unparseable} | {share(stats.unparseable)} |",
        f"| ambiguous SAN | {stats.ambiguous} | {share(stats.ambiguous)} |",
        f"| nothing written | {stats.empty} | {share(stats.empty)} |",
        f"| **asked** | **{stats.asked}** | |",
        "",
    ]
    if result.puzzles is not None:
        lines += [
            "## Puzzles by band",
            "",
            f"Prompts answered without a game prefix: {result.puzzles_blind}.",
            "",
            "| Band | Attempted | Solved | Rate |",
            "|---|---:|---:|---:|",
        ]
        lines += [
            f"| {band.band} | {band.attempted} | {band.solved} | {band.rate * 100:.1f} % |"
            for band in result.puzzles.bands
        ]
        lines.append("")
    if result.elo is not None:
        lines += ["## Games against Stockfish", "", "| Rung | Games | Score |", "|---|---:|---:|"]
        lines += [f"| {rung.name} | {rung.games} | {rung.score:.3f} |" for rung in result.elo.rungs]
        lines.append("")
    if result.notes:
        lines += ["## Notes", ""] + [f"- {note}" for note in result.notes] + [""]
    return "\n".join(lines)


def row_of(result: KarvonenResult) -> WebRow:
    """The row this model contributes to the single results table, flagged as a baseline."""
    elo = result.elo
    interval = (
        [elo.ci_low, elo.ci_high]
        if elo and not elo.separated and elo.ci_low is not None and elo.ci_high is not None
        else None
    )
    return WebRow(
        stage=result.stage,
        params=result.params,
        legality=result.written.legal_rate or None,
        top1=result.top1,
        puzzles=result.puzzles.by_band() if result.puzzles else {},
        elo=elo.elo if elo else None,
        elo_ci=interval,
        elo_lower=elo.elo_lower if elo else None,
        elo_upper=elo.elo_upper if elo else None,
        elo_separated=bool(elo.separated) if elo else False,
        date=result.date,
        baseline=True,
    )


def run_karvonen_suite(
    cfg: EvalConfig,
    stage: str = DEFAULT_STAGE,
    use_cache: bool = True,
    device: str | None = None,
    checkpoint: Path | str | None = None,
) -> tuple[KarvonenResult, ReportPaths]:
    """Evaluate and report, so the row lands in the same table as every other stage."""
    result = evaluate_karvonen(
        cfg, stage=stage, use_cache=use_cache, device=device, checkpoint=checkpoint
    )
    report = write_report_files(
        result.stage,
        render_markdown(result),
        json.loads(result.model_dump_json()),
        paths.resolve(cfg.out_dir),
        paths.resolve(cfg.web_results) if cfg.web_results else None,
        row_of(result),
    )
    return result, report
