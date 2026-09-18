"""Join the sampled positions with ``Lichess/chess-position-evaluations`` (streaming, resumable).

Each remote file ``data_00NN.parquet`` is semi-joined with the local positions by ``fen`` and
written as ``data/evals/part-NN.parquet``; an existing part is skipped, so an interrupted run
resumes where it stopped. ``consolidate`` then keeps one row per FEN with the best line
(deepest lines, mates first, then the best ``cp`` for the side to move) and every line as
``pvs``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.manifest import FileHash, Manifest
from rukh.data.uci import sha256_file
from rukh.paths import resolve

EVAL_FILE = "positions-eval.parquet"
MATE_SCORE = 10_000


class EvalsConfig(BaseConfig):
    """Remote dataset, local positions and the output directory."""

    dataset: str = "Lichess/chess-position-evaluations"
    remote_pattern: str = "hf://datasets/{dataset}/data/data_{index:04d}.parquet"
    n_files: int = Field(default=20, ge=1)
    positions: str = "data/positions/positions.parquet"
    out_dir: str = "data/evals"


def parse_files(spec: str, n_files: int) -> list[int]:
    """``"0-19"``, ``"3"`` or ``"0,2,5-7"`` into a sorted list of file indices."""
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(part))
    bad = [i for i in indices if i < 0 or i >= n_files]
    if bad:
        raise ValueError(f"file index out of range 0-{n_files - 1}: {bad}")
    return sorted(indices)


def _source(cfg: EvalsConfig, index: int) -> str:
    """Remote parquet for file ``index`` (tests monkeypatch this to a local file)."""
    return cfg.remote_pattern.format(dataset=cfg.dataset, index=index)


def part_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"part-{index:02d}.parquet"


def fetch_part(cfg: EvalsConfig, index: int, positions: Path, out_dir: Path) -> tuple[Path, bool]:
    """Semi-join one remote file with the positions; ``(path, fetched)`` (False = existed)."""
    import duckdb

    target = part_path(out_dir, index)
    if target.is_file():
        return target, False
    tmp = target.with_suffix(".parquet.tmp")
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
              SELECT e.fen, e.line, e.depth, e.knodes, e.cp, e.mate
              FROM read_parquet('{_source(cfg, index)}') e
              SEMI JOIN read_parquet('{positions.as_posix()}') p ON e.fen = p.fen4
            ) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()
    tmp.replace(target)
    return target, True


def consolidate(out_dir: Path, positions: Path) -> dict[str, int]:
    """Best line per FEN plus ``pvs``, joined back with the positions' metadata."""
    import duckdb

    parts = (out_dir / "part-*.parquet").as_posix()
    target = out_dir / EVAL_FILE
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            CREATE TEMP TABLE lines AS
            SELECT fen,
                   split_part(line, ' ', 1) AS move,
                   depth, knodes, cp, mate,
                   CASE WHEN split_part(fen, ' ', 2) = 'w' THEN 1 ELSE -1 END AS sign,
                   CASE WHEN mate IS NOT NULL
                        THEN CASE WHEN mate > 0 THEN {MATE_SCORE} - mate
                                  ELSE -{MATE_SCORE} - mate END
                        ELSE cp END AS score_white
            FROM read_parquet('{parts}')
            WHERE line IS NOT NULL AND line <> ''
            """
        )
        con.execute(
            """
            CREATE TEMP TABLE ranked AS
            SELECT *, sign * score_white AS score_mover,
                   row_number() OVER (
                     PARTITION BY fen
                     ORDER BY (mate IS NOT NULL AND sign * mate > 0) DESC,
                              depth DESC, sign * score_white DESC, move
                   ) AS rk
            FROM lines
            """
        )
        con.execute(
            f"""
            COPY (
              SELECT p.fen4 AS fen, p.game_id, p.ply, p.last_move, p.result, p.phase, p.n_seen,
                     b.move AS best_move, b.cp, b.mate, b.depth, b.knodes,
                     pv.pvs
              FROM read_parquet('{positions.as_posix()}') p
              JOIN ranked b ON b.fen = p.fen4 AND b.rk = 1
              JOIN (
                SELECT fen,
                       list(struct_pack(move := move, cp := cp, mate := mate, depth := depth)
                            ORDER BY rk) AS pvs
                FROM ranked GROUP BY fen
              ) pv ON pv.fen = p.fen4
              ORDER BY p.game_id, p.ply
            ) TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        total = con.execute(
            f"SELECT count(*) FROM read_parquet('{positions.as_posix()}')"
        ).fetchone()
        covered = con.execute(
            f"SELECT count(*) FROM read_parquet('{target.as_posix()}')"
        ).fetchone()
        n_lines = con.execute("SELECT count(*) FROM lines").fetchone()
    finally:
        con.close()
    return {
        "positions": int(total[0]) if total else 0,
        "evaluated": int(covered[0]) if covered else 0,
        "lines": int(n_lines[0]) if n_lines else 0,
    }


def run(cfg: EvalsConfig, files: str | None = None) -> Manifest:
    """Fetch the requested parts (skipping existing ones), consolidate and write the manifest."""
    positions = resolve(cfg.positions)
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = parse_files(files or f"0-{cfg.n_files - 1}", cfg.n_files)
    fetched: list[int] = []
    for index in indices:
        _, was_fetched = fetch_part(cfg, index, positions, out_dir)
        if was_fetched:
            fetched.append(index)
    counts = consolidate(out_dir, positions)
    present = sorted(int(p.stem.split("-")[1]) for p in out_dir.glob("part-*.parquet"))
    target = out_dir / EVAL_FILE
    manifest = Manifest(
        dataset=cfg.dataset,
        months=[],
        filters={
            "positions": Path(cfg.positions).as_posix(),
            "files_present": present,
            "files_fetched_now": fetched,
            "n_files": cfg.n_files,
            "coverage": round(counts["evaluated"] / counts["positions"], 4)
            if counts["positions"]
            else 0.0,
            "best_line": "deepest; mate for the mover first; then best cp for the side to move",
        },
        counts=counts,
        files=[FileHash(path=EVAL_FILE, sha256=sha256_file(target), bytes=target.stat().st_size)],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
