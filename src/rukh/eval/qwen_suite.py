"""The same suite, run on a general language model, so the M4 table compares like with like.

Everything here is borrowed: the validation positions of ``rukh.eval.legality``, the puzzle split
of ``rukh.eval.puzzles``, the Stockfish ladder and the bootstrap of ``rukh.eval.elo``. Only the
player differs, which is the whole design -- if the two sides of the comparison went through
different code, the difference in the numbers would be partly the code.

One column exists here and not in the decoder's report, and it is the interesting one. A decoder
whose vocabulary *is* the set of moves can only fail one way: a legal move in the wrong position.
A model writing SAN can also write something that is not a move at all, or a move two pieces could
make and that it failed to disambiguate. Those are counted apart (``QwenStats``) rather than
summed into "illegal", because summing them would hide exactly what the representation costs.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from rukh import paths
from rukh.eval.elo import EloResult, estimate, play_rungs
from rukh.eval.puzzles import PuzzleResult, load_puzzles, run_puzzles
from rukh.eval.qwen_source import QwenPlayer, QwenStats, qwen_source
from rukh.eval.report import WebRow, elo_line, write_report_files
from rukh.eval.suite import EvalConfig, _positions
from rukh.tokenize.uci_vocab import UciTokenizer

if TYPE_CHECKING:
    from rukh.eval.legality import Position

log = logging.getLogger(__name__)

__all__ = ["QwenResult", "evaluate_qwen", "run_qwen_suite"]


class QwenResult(BaseModel):
    """One evaluation of a text model on the decoder's own suite."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    base_model: str
    adapter_dir: str
    params: int
    date: str
    device: str
    positions: int
    written: QwenStats
    """How its answers came out, by failure mode. ``legal_rate`` is the column that faces the
    decoder's "legality without the mask"."""
    top1: float | None = None
    puzzles: PuzzleResult | None = None
    elo: EloResult | None = None
    notes: list[str] = []
    config: dict[str, Any] = {}


def evaluate_qwen(
    adapter_dir: Path | str,
    cfg: EvalConfig,
    stage: str = "qwen3-pgn-qlora",
    use_cache: bool = True,
    device: str | None = None,
) -> QwenResult:
    """Run legality, next-move accuracy, puzzles and Elo on a fine-tuned general model."""
    from rukh.eval.cache import EvalCache, config_sha
    from rukh.train.qwen import load_qwen_adapter

    directory = Path(adapter_dir)
    model, tokenizer = load_qwen_adapter(directory, device=device)
    where = str(next(model.parameters()).device)
    tok = UciTokenizer()
    notes: list[str] = []
    positions: list[Position] = _positions(cfg, tok, notes)
    player = QwenPlayer(model, tokenizer)

    log.info("legality and accuracy over %s positions", min(len(positions), cfg.accuracy_positions))
    scored = positions[: cfg.accuracy_positions]
    hits = 0
    for position in scored:
        board = _board_of(position)
        move, _ = player.propose(board)
        if move is not None and position.target is not None and move.uci() == position.target:
            hits += 1
    written = player.stats.model_copy(deep=True)

    result = QwenResult(
        stage=stage,
        base_model=_base_model(directory),
        adapter_dir=directory.as_posix(),
        params=sum(p.numel() for p in model.parameters()),
        date=datetime.now(UTC).date().isoformat(),
        device=where,
        positions=len(scored),
        written=written,
        top1=hits / len(scored) if scored else None,
        notes=notes,
        config=cfg.model_dump(mode="json"),
    )

    cache = EvalCache(
        paths.resolve(cfg.cache_db) if use_cache else None,
        f"qwen:{stage}",
        enabled=use_cache,
        config_sha=config_sha(cfg.cache_fields()),
    )
    try:
        puzzle_path = paths.resolve(cfg.puzzles)
        if puzzle_path.is_file():
            items = load_puzzles(
                puzzle_path, cfg.puzzles_per_band, seed=cfg.seed, split=cfg.puzzle_split
            )
            log.info("puzzles: %s", len(items))
            result.puzzles = run_puzzles(
                qwen_source(player),
                tok,
                items,
                cache=cache,
                header_elo=cfg.header_elo,
                force_header=cfg.force_header,
            )
        else:
            notes.append(f"puzzles not found at {puzzle_path}: puzzle suite skipped")
        if cfg.elo_games:
            log.info("elo: %s games per rung", cfg.elo_games)
            records = play_rungs(
                None,
                tok,
                cfg.elo_rungs,
                cfg.elo_games,
                cfg.sampling(),
                move_time=cfg.elo_move_time,
                max_plies=cfg.elo_max_plies,
                cache=cache,
                header_elo=cfg.header_elo,
                player=player,
            )
            result.elo = estimate(records, bootstrap=cfg.bootstrap, seed=cfg.seed)
    finally:
        cache.close()
    # `result.written` stays the snapshot taken over the validation positions and nothing else.
    # The player keeps counting through the puzzles and the games, and folding those in would
    # give a legality rate over a different denominator than the decoder's -- which is the one
    # number in this report that has to face the decoder's directly.
    result.notes = notes
    return result


def _board_of(position: Position) -> Any:
    import chess

    board = chess.Board()
    for uci in position.moves:
        board.push(chess.Move.from_uci(uci))
    return board


def _base_model(directory: Path) -> str:
    import json

    config = directory / "adapter_config.json"
    if not config.is_file():
        return "unknown"
    return str(json.loads(config.read_text(encoding="utf-8")).get("base_model_name_or_path", "?"))


def render_markdown(result: QwenResult) -> str:
    """The report, with the failure breakdown the decoder's report has no column for."""
    stats = result.written

    def share(count: int) -> str:
        return f"{100 * count / stats.asked:.2f} %" if stats.asked else "n/a"

    lines = [
        f"# Evaluation of `{result.stage}`",
        "",
        f"- Base model: `{result.base_model}`",
        f"- Adapter: `{result.adapter_dir}`",
        f"- Parameters: {result.params:,}",
        f"- Device: `{result.device}`",
        f"- Date: {result.date}",
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
        "A decoder cannot write a move that does not exist: its vocabulary is the set of moves, "
        "so its only failure is a legal move in the wrong position. A model writing SAN has three "
        "more ways to be wrong, and folding them together would hide what the representation "
        "costs.",
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


def row_of(result: QwenResult) -> WebRow:
    """The row this model contributes to the single results table."""
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
    )


def run_qwen_suite(
    adapter_dir: Path | str,
    cfg: EvalConfig,
    stage: str = "qwen3-pgn-qlora",
    use_cache: bool = True,
    device: str | None = None,
) -> tuple[QwenResult, Any]:
    """Evaluate and report, so the row lands in the same table as every other stage."""
    import json

    result = evaluate_qwen(adapter_dir, cfg, stage=stage, use_cache=use_cache, device=device)
    report = write_report_files(
        result.stage,
        render_markdown(result),
        json.loads(result.model_dump_json()),
        paths.resolve(cfg.out_dir),
        paths.resolve(cfg.web_results) if cfg.web_results else None,
        row_of(result),
    )
    return result, report
