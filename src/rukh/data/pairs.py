"""DPO pairs from the multi-PV evaluations: best first move versus a clearly worse one.

The lines are ranked with ``rukh.data.scoring``, the same ordering and mate convention the
``evals`` consolidation uses, so ``chosen`` is always the consolidated ``best_move``. The
margin is measured from the side to move (for Black a lower ``cp`` is better). Both moves are
checked for legality with python-chess. One pair per FEN, balanced by phase.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.manifest import FileHash, Manifest
from rukh.data.scoring import (
    BEST_LINE_DOC,
    MATE_SCORE,
    line_rank_key,
    score_mover,
    score_white,
)
from rukh.data.uci import sha256_file
from rukh.paths import resolve

LOGGER = logging.getLogger(__name__)
PAIRS_FILE = "pairs.parquet"
PHASES = ("opening", "middlegame", "endgame")
BATCH_ROWS = 50_000


class PairsConfig(BaseConfig):
    """Input evaluations, margin and the per-phase cap."""

    positions_eval: str = "data/evals/positions-eval.parquet"
    out_dir: str = "data/pairs"
    min_delta_cp: int = Field(default=100, ge=1)
    max_per_phase: int = Field(default=200_000, ge=1)
    seed: int = 42


def rank_lines(fen: str, pvs: list[dict[str, object]]) -> list[tuple[str, int, int]]:
    """``(move, score_white, score_mover)`` for every scored line, best line first."""
    turn = fen.split()[1]
    scored: list[tuple[tuple[bool, int, int, str], str, int, int]] = []
    for pv in pvs:
        move = pv.get("move")
        cp, mate, depth = pv.get("cp"), pv.get("mate"), pv.get("depth")
        white = score_white(cp, mate)  # type: ignore[arg-type]
        if not move or white is None:
            continue
        key = line_rank_key(cp, mate, depth, str(move), turn)  # type: ignore[arg-type]
        scored.append((key, str(move), white, score_mover(cp, mate, turn)))  # type: ignore[arg-type]
    scored.sort(key=lambda item: item[0])
    return [(move, white, mover) for _, move, white, mover in scored]


def make_pair(
    fen: str, pvs: list[dict[str, object]], min_delta_cp: int
) -> dict[str, object] | None:
    """``chosen``/``rejected`` for one FEN or ``None`` when no line is bad enough or illegal."""
    import chess

    scored = rank_lines(fen, pvs)
    if len(scored) < 2:
        return None
    chosen, cp_chosen, best_mover = scored[0]
    rejected: tuple[str, int, int] | None = None
    for move, white, mover in scored[1:]:
        if move != chosen and best_mover - mover >= min_delta_cp:
            rejected = (move, white, mover)
            break
    if rejected is None:
        return None
    try:
        board = chess.Board(f"{fen} 0 1")
    except ValueError:
        return None
    legal = {m.uci() for m in board.legal_moves}
    if chosen not in legal or rejected[0] not in legal:
        return None
    return {
        "fen": fen,
        "chosen": chosen,
        "rejected": rejected[0],
        "cp_chosen": cp_chosen,
        "cp_rejected": rejected[1],
    }


def build_pairs(evals: Path, min_delta_cp: int) -> pl.DataFrame:
    """One pair per FEN, reading the evaluations parquet batch by batch."""
    reader = pq.ParquetFile(evals)
    rows: list[dict[str, object]] = []
    for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=["fen", "phase", "pvs"]):
        records = zip(
            batch.column("fen").to_pylist(),
            batch.column("phase").to_pylist(),
            batch.column("pvs").to_pylist(),
            strict=True,
        )
        for fen, ph, pvs in records:
            pair = make_pair(fen, pvs, min_delta_cp)
            if pair is not None:
                pair["phase"] = ph
                rows.append(pair)
    schema = {
        "fen": pl.String,
        "chosen": pl.String,
        "rejected": pl.String,
        "cp_chosen": pl.Int32,
        "cp_rejected": pl.Int32,
        "phase": pl.String,
    }
    return pl.DataFrame(rows, schema=schema)


def phase_counts(frame: pl.DataFrame) -> dict[str, int]:
    """Rows per phase, always with the three keys."""
    return {ph: int(frame.filter(pl.col("phase") == ph).height) for ph in PHASES}


def balance(frame: pl.DataFrame, max_per_phase: int, seed: int) -> pl.DataFrame:
    """Same number of pairs per phase: the smallest phase count, capped at ``max_per_phase``."""
    counts = phase_counts(frame)
    n = min(min(counts.values()), max_per_phase)
    if n == 0:
        return frame.clear()
    parts = [
        frame.filter(pl.col("phase") == ph).sample(n=n, seed=seed, shuffle=True) for ph in PHASES
    ]
    return pl.concat(parts)


def run(cfg: PairsConfig) -> Manifest:
    """Build, balance and write ``out_dir/pairs.parquet`` plus the manifest."""
    evals = resolve(cfg.positions_eval)
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = build_pairs(evals, cfg.min_delta_cp)
    candidates = phase_counts(raw)
    empty = [ph for ph, n in candidates.items() if n == 0]
    if empty:
        LOGGER.warning(
            "no DPO candidates for %s: the balanced output is empty (candidates %s). "
            "Check the coverage of %s.",
            ", ".join(empty),
            candidates,
            evals,
        )
    balanced = balance(raw, cfg.max_per_phase, cfg.seed)
    target = out_dir / PAIRS_FILE
    balanced.write_parquet(target.as_posix(), compression="zstd")
    counts = phase_counts(balanced)
    counts["candidates"] = int(raw.height)
    manifest = Manifest(
        dataset="Lichess/chess-position-evaluations",
        months=[],
        filters={
            "min_delta_cp": cfg.min_delta_cp,
            "max_per_phase": cfg.max_per_phase,
            "seed": cfg.seed,
            "mate_score": MATE_SCORE,
            "cp_point_of_view": "white",
            "best_line": BEST_LINE_DOC,
            "legality": "python-chess",
            "candidates_by_phase": candidates,
            "empty_phases": empty,
        },
        counts=counts,
        files=[FileHash(path=PAIRS_FILE, sha256=sha256_file(target), bytes=target.stat().st_size)],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
