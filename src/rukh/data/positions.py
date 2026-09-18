"""Positions from the UCI games: normalized 4-field FEN, phase, deduplicated with a count.

``fen4`` is ``pieces turn castling en-passant`` (the key of
``Lichess/chess-position-evaluations``); move counters are dropped so the same position from
different games collapses into one row with ``n_seen`` occurrences. The first occurrence wins
the ``game_id``/``ply``/``last_move``/``result`` columns.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.manifest import FileHash, Manifest
from rukh.data.parallel import run_batches
from rukh.data.uci import resolve_workers, sha256_file
from rukh.paths import resolve

OPENING_MAX_PLY = 10
MIDDLEGAME_MIN_PIECES = 14
BATCH_GAMES = 5_000
POSITIONS_FILE = "positions.parquet"
PARTS_DIR = "parts"
POSITION_SCHEMA = pa.schema(
    [
        ("seq", pa.int64()),
        ("fen4", pa.string()),
        ("game_id", pa.int64()),
        ("ply", pa.int16()),
        ("last_move", pa.string()),
        ("result", pa.string()),
        ("phase", pa.string()),
    ]
)


class PositionsConfig(BaseConfig):
    """Which games to walk and how many distinct positions to keep."""

    uci_dir: str = "data/uci"
    month: str = "2025-01"
    out_dir: str = "data/positions"
    n_games: int = Field(default=300_000, ge=1)
    max_positions: int = Field(default=5_000_000, ge=1)
    workers: int = Field(default=0, ge=0)


def fen4(board: object) -> str:
    """First four FEN fields (python-chess writes the en-passant square only when capturable)."""
    return " ".join(board.fen().split()[:4])  # type: ignore[attr-defined]


def phase(ply: int, n_pieces: int) -> str:
    """``opening`` up to ply 10, ``middlegame`` with 14+ pieces afterwards, else ``endgame``."""
    if ply <= OPENING_MAX_PLY:
        return "opening"
    return "middlegame" if n_pieces >= MIDDLEGAME_MIN_PIECES else "endgame"


def walk_game(game_id: int, uci: str, result: str) -> Iterator[tuple[str, int, int, str, str, str]]:
    """Yield ``(fen4, game_id, ply, last_move, result, phase)`` after every move."""
    import chess

    board = chess.Board()
    for ply, token in enumerate(uci.split(), start=1):
        move = chess.Move.from_uci(token)
        board.push(move)
        yield fen4(board), game_id, ply, token, result, phase(ply, len(board.piece_map()))


def walk_rows(rows: list[tuple[int, str, str]]) -> dict[str, list[object]]:
    """Walk a batch of ``(game_id, uci, result)``; module-level for the spawn pool."""
    out: dict[str, list[object]] = {name: [] for name in POSITION_SCHEMA.names if name != "seq"}
    for game_id, uci, result in rows:
        for fen, gid, ply, last_move, res, ph in walk_game(game_id, uci, result):
            out["fen4"].append(fen)
            out["game_id"].append(gid)
            out["ply"].append(ply)
            out["last_move"].append(last_move)
            out["result"].append(res)
            out["phase"].append(ph)
    return out


def iter_games(games: Path, n_games: int) -> Iterator[list[tuple[int, str, str]]]:
    """Batches of ``(game_id, uci, result)``, read from the parquet in row groups."""
    reader = pq.ParquetFile(games)
    remaining = n_games
    for batch in reader.iter_batches(batch_size=BATCH_GAMES, columns=["game_id", "uci", "result"]):
        if remaining <= 0:
            return
        rows = list(
            zip(
                batch.column("game_id").to_pylist(),
                batch.column("uci").to_pylist(),
                batch.column("result").to_pylist(),
                strict=True,
            )
        )[:remaining]
        remaining -= len(rows)
        yield rows


def write_parts(games: Path, parts_dir: Path, cfg: PositionsConfig) -> int:
    """Walk the games into ``parts_dir/part-NNNN.parquet`` (all positions, not deduplicated)."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    for old in parts_dir.glob("part-*.parquet"):
        old.unlink()
    workers = resolve_workers(cfg.workers)
    batches = iter_games(games, cfg.n_games)
    return _write_parts(parts_dir, run_batches(walk_rows, batches, workers), 0)


def _write_parts(parts_dir: Path, results: Iterator[dict[str, list[object]]], seq: int) -> int:
    for index, columns in enumerate(results):
        n = len(columns["fen4"])
        table = pa.table({"seq": list(range(seq, seq + n)), **columns}, schema=POSITION_SCHEMA)
        pq.write_table(table, parts_dir / f"part-{index:04d}.parquet", compression="zstd")
        seq += n
    return seq


def dedupe(parts_dir: Path, out: Path, max_positions: int) -> int:
    """Group the parts by ``fen4`` (first occurrence wins, ``n_seen`` counted) with DuckDB."""
    import duckdb

    glob = (parts_dir / "part-*.parquet").as_posix()
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
              SELECT fen4,
                     arg_min(game_id, seq) AS game_id,
                     arg_min(ply, seq) AS ply,
                     arg_min(last_move, seq) AS last_move,
                     arg_min(result, seq) AS result,
                     arg_min(phase, seq) AS phase,
                     count(*) AS n_seen
              FROM read_parquet('{glob}')
              GROUP BY fen4
              ORDER BY min(seq)
              LIMIT {int(max_positions)}
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        row = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()
    finally:
        con.close()
    return int(row[0]) if row else 0


def month_parquet(uci_dir: Path, month: str) -> Path:
    year, mm = month.split("-")
    return uci_dir / f"year={year}" / f"month={mm}" / "games.parquet"


def run(cfg: PositionsConfig) -> Manifest:
    """Walk up to ``n_games`` of ``month`` and write ``out_dir/positions.parquet`` + manifest."""
    games = month_parquet(resolve(cfg.uci_dir), cfg.month)
    out_dir = resolve(cfg.out_dir)
    parts_dir = out_dir / PARTS_DIR
    n_positions = write_parts(games, parts_dir, cfg)
    out = out_dir / POSITIONS_FILE
    n_distinct = dedupe(parts_dir, out, cfg.max_positions)
    manifest = Manifest(
        dataset="Lichess/standard-chess-games",
        months=[cfg.month],
        filters={
            "n_games": cfg.n_games,
            "max_positions": cfg.max_positions,
            "phase": {
                "opening_max_ply": OPENING_MAX_PLY,
                "middlegame_min_pieces": MIDDLEGAME_MIN_PIECES,
            },
        },
        counts={"positions": n_positions, "distinct": n_distinct},
        files=[FileHash(path=POSITIONS_FILE, sha256=sha256_file(out), bytes=out.stat().st_size)],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
