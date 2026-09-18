"""Puzzle splits by difficulty band from ``Lichess/chess-puzzles`` (deterministic by id)."""

from __future__ import annotations

import hashlib

import polars as pl
from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.manifest import FileHash, Manifest
from rukh.data.uci import sha256_file
from rukh.paths import resolve

PUZZLES_FILE = "puzzles.parquet"
BANDS: list[tuple[str, int, int | None]] = [
    ("1000-1500", 1000, 1500),
    ("1500-2000", 1500, 2000),
    ("2000+", 2000, None),
]


class PuzzlesConfig(BaseConfig):
    """Remote dataset, quality filters and split sizes."""

    dataset: str = "Lichess/chess-puzzles"
    remote_glob: str = "hf://datasets/{dataset}/**/*.parquet"
    out_dir: str = "data/puzzles"
    max_rating_deviation: int = Field(default=100, ge=0)
    min_plays: int = Field(default=100, ge=0)
    n_test: int = Field(default=2_000, ge=1)
    n_train: int = Field(default=50_000, ge=1)
    seed: int = 42


def band(rating: int) -> str | None:
    """Difficulty band for ``rating``; ``None`` below 1000."""
    for name, lo, hi in BANDS:
        if rating >= lo and (hi is None or rating < hi):
            return name
    return None


def _source(cfg: PuzzlesConfig) -> str:
    """Remote parquet glob (tests monkeypatch this to a local file)."""
    return cfg.remote_glob.format(dataset=cfg.dataset)


def load_filtered(cfg: PuzzlesConfig) -> pl.DataFrame:
    """Read the puzzles that pass the quality filters with DuckDB (predicate pushdown)."""
    import duckdb

    con = duckdb.connect()
    try:
        table = con.execute(
            f"""
            SELECT PuzzleId AS puzzle_id, FEN AS fen, Moves AS moves, Rating AS rating,
                   Themes AS themes
            FROM read_parquet('{_source(cfg)}')
            WHERE RatingDeviation <= {cfg.max_rating_deviation}
              AND NbPlays >= {cfg.min_plays}
              AND Rating >= 1000
            """
        ).arrow()
    finally:
        con.close()
    return pl.from_arrow(table)  # type: ignore[return-value]


def split_key(puzzle_id: str, seed: int) -> int:
    """Stable 64-bit order key: ``blake2b(seed:puzzle_id)``."""
    digest = hashlib.blake2b(f"{seed}:{puzzle_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def split_frame(frame: pl.DataFrame, cfg: PuzzlesConfig) -> pl.DataFrame:
    """Assign ``band`` and ``split``: per band, first ``n_test`` by key then up to ``n_train``."""
    keyed = frame.with_columns(
        pl.col("rating").map_elements(band, return_dtype=pl.String).alias("band"),
        pl.col("puzzle_id")
        .map_elements(lambda p: split_key(p, cfg.seed), return_dtype=pl.UInt64)
        .alias("key"),
        pl.col("themes").cast(pl.List(pl.String)),
    ).filter(pl.col("band").is_not_null())
    keyed = keyed.sort(["band", "key"]).with_columns(
        pl.int_range(pl.len()).over("band").alias("rank")
    )
    return (
        keyed.with_columns(
            pl.when(pl.col("rank") < cfg.n_test)
            .then(pl.lit("test"))
            .when(pl.col("rank") < cfg.n_test + cfg.n_train)
            .then(pl.lit("train"))
            .otherwise(None)
            .alias("split")
        )
        .filter(pl.col("split").is_not_null())
        .select("puzzle_id", "fen", "moves", "rating", "themes", "band", "split")
    )


def run(cfg: PuzzlesConfig) -> Manifest:
    """Download, filter, split and write ``out_dir/puzzles.parquet`` plus the manifest."""
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = split_frame(load_filtered(cfg), cfg)
    target = out_dir / PUZZLES_FILE
    frame.write_parquet(target.as_posix(), compression="zstd")
    counts = {
        f"{b}/{s}": int(n)
        for b, s, n in frame.group_by("band", "split").len().sort("band", "split").iter_rows()
    }
    manifest = Manifest(
        dataset=cfg.dataset,
        months=[],
        filters={
            "max_rating_deviation": cfg.max_rating_deviation,
            "min_plays": cfg.min_plays,
            "bands": [b for b, _, _ in BANDS],
            "n_test": cfg.n_test,
            "n_train": cfg.n_train,
            "seed": cfg.seed,
            "split_by": "blake2b(seed:PuzzleId)",
        },
        counts=counts,
        files=[
            FileHash(path=PUZZLES_FILE, sha256=sha256_file(target), bytes=target.stat().st_size)
        ],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
