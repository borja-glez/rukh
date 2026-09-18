"""Elo-balanced sample: up to ``n_per_bin`` games per 100-Elo bin of the average rating."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.db import DuckDbConfig, connect
from rukh.data.manifest import FileHash, Manifest
from rukh.data.uci import month_dirs, sha256_file
from rukh.paths import resolve

GAMES_FILE = "games.parquet"


class EloBinsConfig(BaseConfig):
    """Source months, per-bin cap and seed."""

    uci_dir: str = "data/uci"
    out_dir: str = "data/elo-bins"
    n_per_bin: int = Field(default=50_000, ge=1)
    seed: int = 42
    duckdb: DuckDbConfig = Field(default_factory=DuckDbConfig)


def sample_bins(
    sources: list[Path],
    target: Path,
    n_per_bin: int,
    seed: int,
    duckdb_cfg: DuckDbConfig | None = None,
) -> dict[str, int]:
    """DuckDB: ``bin = avg(white_elo, black_elo) // 100 * 100``; keep ``n_per_bin`` by hash.

    The seed shifts the hash, not the id: ``hash(game_id + seed)`` overflowed BIGINT for ids
    near the top of the range.
    """
    files = ", ".join(f"'{p.as_posix()}'" for p in sources)
    con = connect(duckdb_cfg)
    try:
        con.execute(
            f"""
            COPY (
              SELECT * EXCLUDE (rk)
              FROM (
                SELECT *,
                       row_number() OVER (PARTITION BY bin ORDER BY hash(game_id) + {seed}) AS rk
                FROM (
                  SELECT *, ((white_elo + black_elo) // 2) // 100 * 100 AS bin
                  FROM read_parquet([{files}], hive_partitioning = false)
                )
              )
              WHERE rk <= {int(n_per_bin)}
              ORDER BY bin, rk
            ) TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        rows = con.execute(
            f"SELECT bin, count(*) FROM read_parquet('{target.as_posix()}') "
            "GROUP BY bin ORDER BY bin"
        ).fetchall()
    finally:
        con.close()
    return {str(int(b)): int(n) for b, n in rows}


def run(cfg: EloBinsConfig) -> Manifest:
    """Sample every month under ``uci_dir`` and write ``out_dir/games.parquet`` + manifest."""
    months = month_dirs(resolve(cfg.uci_dir))
    if not months:
        raise FileNotFoundError(f"no games under {resolve(cfg.uci_dir)}")
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / GAMES_FILE
    counts = sample_bins([p for _, p in months], target, cfg.n_per_bin, cfg.seed, cfg.duckdb)
    manifest = Manifest(
        dataset="Lichess/standard-chess-games",
        months=[m for m, _ in months],
        filters={
            "bin": "((white_elo + black_elo) // 2) // 100 * 100",
            "n_per_bin": cfg.n_per_bin,
            "seed": cfg.seed,
        },
        counts=counts,
        files=[FileHash(path=GAMES_FILE, sha256=sha256_file(target), bytes=target.stat().st_size)],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
