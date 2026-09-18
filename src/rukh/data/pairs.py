"""DPO pairs from the multi-PV evaluations: best first move versus a clearly worse one.

``cp`` and ``mate`` come from White's point of view; the comparison is made from the side to
move (for Black a lower ``cp`` is better). A mate counts as ``+/-10000`` centipawns with the
sign of the side that mates. Both moves are checked for legality with python-chess. One pair
per FEN, balanced by phase.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.manifest import FileHash, Manifest
from rukh.data.uci import sha256_file
from rukh.paths import resolve

PAIRS_FILE = "pairs.parquet"
MATE_SCORE = 10_000
PHASES = ("opening", "middlegame", "endgame")
BATCH_ROWS = 50_000


class PairsConfig(BaseConfig):
    """Input evaluations, margin and the per-phase cap."""

    positions_eval: str = "data/evals/positions-eval.parquet"
    out_dir: str = "data/pairs"
    min_delta_cp: int = Field(default=100, ge=1)
    max_per_phase: int = Field(default=200_000, ge=1)
    seed: int = 42


def score_white(cp: int | None, mate: int | None) -> int | None:
    """Centipawns from White's view; mates map to ``+/-10000``."""
    if mate is not None:
        return MATE_SCORE if mate > 0 else -MATE_SCORE
    return None if cp is None else int(cp)


def score_mover(cp: int | None, mate: int | None, turn: str) -> int | None:
    """Score from the side to move: unchanged for White, negated for Black."""
    white = score_white(cp, mate)
    if white is None:
        return None
    return white if turn == "w" else -white


def make_pair(
    fen: str, pvs: list[dict[str, object]], min_delta_cp: int
) -> dict[str, object] | None:
    """``chosen``/``rejected`` for one FEN or ``None`` when no line is bad enough or illegal."""
    import chess

    turn = fen.split()[1]
    scored: list[tuple[int, str, int]] = []
    for pv in pvs:
        mover = score_mover(pv.get("cp"), pv.get("mate"), turn)  # type: ignore[arg-type]
        move = pv.get("move")
        if mover is None or not move:
            continue
        white = score_white(pv.get("cp"), pv.get("mate"))  # type: ignore[arg-type]
        scored.append((mover, str(move), int(white)))  # type: ignore[arg-type]
    if len(scored) < 2:
        return None
    scored.sort(key=lambda t: -t[0])
    best_mover, chosen, cp_chosen = scored[0]
    rejected: tuple[int, str, int] | None = None
    for mover, move, white in scored[1:]:
        if move != chosen and best_mover - mover >= min_delta_cp:
            rejected = (mover, move, white)
            break
    if rejected is None:
        return None
    try:
        board = chess.Board(f"{fen} 0 1")
    except ValueError:
        return None
    legal = {m.uci() for m in board.legal_moves}
    if chosen not in legal or rejected[1] not in legal:
        return None
    return {
        "fen": fen,
        "chosen": chosen,
        "rejected": rejected[1],
        "cp_chosen": cp_chosen,
        "cp_rejected": rejected[2],
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


def balance(frame: pl.DataFrame, max_per_phase: int, seed: int) -> pl.DataFrame:
    """Same number of pairs per phase: the smallest phase count, capped at ``max_per_phase``."""
    counts = {ph: int(frame.filter(pl.col("phase") == ph).height) for ph in PHASES}
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
    balanced = balance(raw, cfg.max_per_phase, cfg.seed)
    target = out_dir / PAIRS_FILE
    balanced.write_parquet(target.as_posix(), compression="zstd")
    counts = {ph: int(balanced.filter(pl.col("phase") == ph).height) for ph in PHASES}
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
            "legality": "python-chess",
        },
        counts=counts,
        files=[FileHash(path=PAIRS_FILE, sha256=sha256_file(target), bytes=target.stat().st_size)],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
