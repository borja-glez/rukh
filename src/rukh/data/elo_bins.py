"""Elo-balanced sample: up to ``n_per_bin`` games per band of the average rating.

The first version of this file read one directory of months and binned by hundreds, which is all
the 1800+ corpus could support. It turned out to be the reason "play like 1500" did nothing: the
whole project had been fetched with ``min_elo: 1800``, so ``data/elo-bins`` started at the 1800
bin and the header tokens below it never received a gradient. Conditioning on a weak rating
showed the model a vector that was still at its initialisation.

Filling the axis needs two things this module now has. **Several sources**, because the low band
lives in its own corpus (``data/uci-low``) and must not be mixed into ``data/uci``, which is what
every published model trained on. And a **band width**, because a flat sample over 100-Elo bins
wastes the budget at the crowded end and starves the thin one: the top bins are naturally small
(2 316 games at 2800 against 50 000 at 1800), so the band the sample is balanced over is a
parameter and the manifest publishes what each one really got.

"Flat" is the point. A corpus whose ratings are distributed like Lichess teaches the model that
`<w1500>` is rare and `<w2000>` is normal; a corpus with the same number of games per band
teaches it what each header *means*, which is what conditioning needs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from rukh.config import BaseConfig
from rukh.data.db import DuckDbConfig, connect
from rukh.data.manifest import FileHash, Manifest
from rukh.data.uci import month_dirs, sha256_file
from rukh.paths import resolve

GAMES_FILE = "games.parquet"


class EloBinsConfig(BaseConfig):
    """Source trees, band width, per-band cap and seed."""

    sources: list[str] = ["data/uci"]
    """Every tree of ``year=YYYY/month=MM/games.parquet`` to sample from, in order.

    A list and not one directory because the axis is split across corpora: the 1800+ games and
    the 1000-1799 ones were fetched separately and live apart on purpose, so that
    ``data/uci`` keeps meaning exactly what the published models trained on."""
    out_dir: str = "data/elo-bins"
    bin_width: int = Field(default=100, ge=50)
    """Width of a band in Elo. The sample is flat over *these*, so widening them trades
    resolution along the axis for more games in the thin bands at the top."""
    min_bin: int | None = None
    """Lowest band kept, inclusive. ``None`` keeps everything the sources have."""
    max_bin: int | None = None
    """Highest band kept, inclusive of its whole width."""
    n_per_bin: int = Field(default=50_000, ge=1)
    seed: int = 42
    duckdb: DuckDbConfig = Field(default_factory=DuckDbConfig)

    @model_validator(mode="after")
    def _check(self) -> EloBinsConfig:
        if not self.sources:
            raise ValueError("sources must name at least one directory")
        if self.min_bin is not None and self.max_bin is not None and self.max_bin < self.min_bin:
            raise ValueError(f"max_bin {self.max_bin} is below min_bin {self.min_bin}")
        return self


def _bin_expression(bin_width: int) -> str:
    return f"((white_elo + black_elo) // 2) // {bin_width} * {bin_width}"


def sample_bins(
    sources: list[Path],
    target: Path,
    n_per_bin: int,
    seed: int,
    duckdb_cfg: DuckDbConfig | None = None,
    bin_width: int = 100,
    min_bin: int | None = None,
    max_bin: int | None = None,
) -> dict[str, int]:
    """DuckDB: ``bin = avg(white_elo, black_elo) // width * width``; keep ``n_per_bin`` by hash.

    The seed shifts the hash, not the id: ``hash(game_id + seed)`` overflowed BIGINT for ids
    near the top of the range.
    """
    files = ", ".join(f"'{p.as_posix()}'" for p in sources)
    where = ["true"]
    if min_bin is not None:
        where.append(f"bin >= {int(min_bin)}")
    if max_bin is not None:
        where.append(f"bin <= {int(max_bin)}")
    band_filter = " AND ".join(where)
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
                  SELECT *, {_bin_expression(bin_width)} AS bin
                  FROM read_parquet([{files}], hive_partitioning = false, union_by_name = true)
                )
                WHERE {band_filter}
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
    """Sample every month under each source and write ``out_dir/games.parquet`` + manifest."""
    months: list[tuple[str, Path]] = []
    for source in cfg.sources:
        found = month_dirs(resolve(source))
        if not found:
            raise FileNotFoundError(f"no games under {resolve(source)}")
        months.extend(found)
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / GAMES_FILE
    counts = sample_bins(
        [p for _, p in months],
        target,
        cfg.n_per_bin,
        cfg.seed,
        cfg.duckdb,
        bin_width=cfg.bin_width,
        min_bin=cfg.min_bin,
        max_bin=cfg.max_bin,
    )
    manifest = Manifest(
        dataset="Lichess/standard-chess-games",
        months=sorted({month for month, _ in months}),
        filters={
            "bin": _bin_expression(cfg.bin_width),
            "sources": list(cfg.sources),
            "bin_width": cfg.bin_width,
            "min_bin": cfg.min_bin,
            "max_bin": cfg.max_bin,
            "n_per_bin": cfg.n_per_bin,
            "seed": cfg.seed,
        },
        counts=counts,
        files=[FileHash(path=GAMES_FILE, sha256=sha256_file(target), bytes=target.stat().st_size)],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
