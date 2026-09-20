"""Fetch a filtered slice of ``Lichess/standard-chess-games`` with DuckDB over ``hf://``.

Only the games that pass the filters are materialized locally (predicate pushdown on the
remote parquet files). The ply count (``min_plies``) is not applied here: it is enforced in P1
once ``movetext`` has been converted to UCI with python-chess, because counting plies from SAN
with comments (``%clk``, ``%eval``) in SQL is unreliable. The manifest therefore records it as
``min_plies_deferred``. Variant exclusion by ``Event`` is a coarse first pass that P1 refines.

The months are read **shard by shard, downloaded first**, not streamed over ``hf://``. Streaming
was the original design and it is the obvious one: DuckDB reads the remote parquet directly and
only the rows that pass the filter ever land on disk. It stopped working. A month is 72 files of
about a gigabyte, and reading them remotely means thousands of small range requests against one
host, which earns an ``HTTP 429`` partway through and then a retry schedule that doubles until
the throughput is measured in kilobytes. Measured on 2026-09-20: streaming stalled at 0.02 MB/s
after fifty minutes with nothing written, while downloading the same shard whole ran at 14.8 MB/s.

So each shard is fetched once, with ``huggingface_hub`` (which caches, resumes and backs off
properly), filtered locally into a part file and deleted again. Peak disk is one shard, the work
resumes where it stopped, and ``limit`` now stops the download early instead of only trimming the
result -- for a slice of the rating range that is the difference between 8 shards and 72.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rukh import paths
from rukh.config import BaseConfig
from rukh.data.db import DuckDbConfig, connect
from rukh.data.manifest import FileHash, Manifest

log = logging.getLogger(__name__)

MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


class FetchConfig(BaseConfig):
    """What to fetch and how to filter it."""

    dataset: str = "Lichess/standard-chess-games"
    months: list[str] = Field(min_length=1)
    min_elo: int = Field(default=1800, ge=0)
    max_elo: int | None = Field(default=None, ge=0)
    """Upper bound on both ratings, so a second corpus can cover the band the first one left out.

    The project's whole corpus was fetched with ``min_elo: 1800`` and no ceiling, which means the
    Elo header tokens below ``<w1800>`` were never trained: conditioning the model on 1500 shows it
    a vector that is still at its initialisation. Filling that band needs a fetch that is bounded
    above as well, and a bound that is applied to *both* players for the same reason ``min_elo``
    is -- a 1200 against a 2400 is not a 1200-rated game, it is a mismatch."""
    min_base_seconds: int = Field(default=180, ge=0)
    terminations: list[str] = Field(default=["Normal", "Time forfeit"], min_length=1)
    min_plies: int = Field(default=20, ge=0)
    exclude_variants: bool = True
    out_dir: str = "data/raw"
    limit: int | None = Field(default=None, ge=1)
    """Rows to keep per month. It also stops the download: once the parts hold this many games
    the remaining shards are never fetched."""
    cache_dir: str = "data/hf-cache"
    """Where downloaded shards are parked before being filtered, and deleted from afterwards."""
    keep_shards: bool = False
    """Keep the downloaded shards instead of deleting each one once it has been filtered.
    72 GB per month, so it is off; useful when the same month is going to be re-filtered."""
    duckdb: DuckDbConfig = Field(default_factory=DuckDbConfig)

    @field_validator("months")
    @classmethod
    def _check_months(cls, months: list[str]) -> list[str]:
        for month in months:
            if not MONTH_RE.match(month):
                raise ValueError(f"month {month!r} must look like YYYY-MM")
        return months

    @model_validator(mode="after")
    def _check_elo_band(self) -> FetchConfig:
        if self.max_elo is not None and self.max_elo < self.min_elo:
            raise ValueError(f"max_elo {self.max_elo} is below min_elo {self.min_elo}")
        return self


class FetchPlan(BaseModel):
    """Everything ``run`` would do, without doing it.

    ``out_dir`` is the absolute output directory; ``out_paths`` and ``manifest_path`` are
    relative to it, in POSIX form, so plans and manifests do not leak machine paths.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    months: list[str]
    out_dir: str
    out_paths: list[str]
    manifest_path: str


def _split_month(month: str) -> tuple[str, str]:
    match = MONTH_RE.match(month)
    if match is None:
        raise ValueError(f"month {month!r} must look like YYYY-MM")
    return match.group(1), match.group(2)


def _source(cfg: FetchConfig, month: str) -> str:
    year, mm = _split_month(month)
    return f"hf://datasets/{cfg.dataset}/data/year={year}/month={mm}/*.parquet"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def where_clause(cfg: FetchConfig) -> str:
    """The filter itself, one condition per line: the only thing that decides what is kept.

    ``TRY_CAST`` drops rows whose base time is not an integer (correspondence games export
    ``TimeControl = '-'``). It is built once and used twice -- in the query the dry run prints
    and against every downloaded shard -- so what is documented is what actually runs.
    """
    conditions = [
        f"WhiteElo >= {cfg.min_elo} AND BlackElo >= {cfg.min_elo}",
        f"TRY_CAST(split_part(TimeControl, '+', 1) AS INTEGER) >= {cfg.min_base_seconds}",
        "Termination IN (" + ", ".join(_sql_string(t) for t in cfg.terminations) + ")",
    ]
    if cfg.max_elo is not None:
        conditions.insert(1, f"WhiteElo <= {cfg.max_elo} AND BlackElo <= {cfg.max_elo}")
    if cfg.exclude_variants:
        conditions.append("Event NOT ILIKE '%variant%'")
    return "\n    AND ".join(conditions)


def build_query(cfg: FetchConfig) -> str:
    """The whole selection as one readable statement, for the dry run and the record.

    ``hive_partitioning = false`` stops DuckDB from injecting ``year`` and ``month`` columns from
    the path, which would collide with the explicit ``month`` alias.
    """
    where = where_clause(cfg)
    selects = [
        "SELECT *, "
        + _sql_string(month)
        + " AS month\n"
        + f"FROM read_parquet({_sql_string(_source(cfg, month))}, hive_partitioning = false)\n"
        + f"WHERE {where}"
        for month in cfg.months
    ]
    sql = "\nUNION ALL\n".join(selects)
    if cfg.limit is not None:
        sql += f"\nLIMIT {cfg.limit}"
    return sql


def _out_dir(cfg: FetchConfig) -> Path:
    out = Path(cfg.out_dir)
    return out if out.is_absolute() else paths.root() / out


def plan(cfg: FetchConfig) -> FetchPlan:
    """Describe the query, months, output directory, relative output files and manifest."""
    out_dir = _out_dir(cfg)
    out_paths = []
    for month in cfg.months:
        year, mm = _split_month(month)
        target = out_dir / f"year={year}" / f"month={mm}" / "games.parquet"
        out_paths.append(target.relative_to(out_dir).as_posix())
    return FetchPlan(
        query=build_query(cfg),
        months=list(cfg.months),
        out_dir=out_dir.as_posix(),
        out_paths=out_paths,
        manifest_path=(out_dir / "manifest.json").relative_to(out_dir).as_posix(),
    )


def _is_empty(path: Path) -> bool:
    """Whether the month still has to be fetched: missing, or the stub a failed COPY leaves."""
    return not path.is_file() or path.stat().st_size == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def month_shards(cfg: FetchConfig, month: str) -> list[str]:
    """Repository paths of the parquet shards of one month, in order."""
    from huggingface_hub import HfApi

    year, mm = _split_month(month)
    prefix = f"data/year={year}/month={mm}/"
    api = HfApi()
    files = api.list_repo_files(cfg.dataset, repo_type="dataset")
    return sorted(name for name in files if name.startswith(prefix) and name.endswith(".parquet"))


def _download(cfg: FetchConfig, shard: str) -> Path:
    """One shard as a real file on disk, resumed if a previous run was cut off partway.

    ``local_dir`` and not ``cache_dir``. The Hub cache stores the bytes under ``blobs/`` and hands
    back a link into ``snapshots/``, so deleting what this returns frees nothing: the first run
    that way left 5.7 GB of shards behind after claiming to have cleaned up. With ``local_dir``
    the path *is* the file, and ``unlink`` means what it says.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            cfg.dataset,
            shard,
            repo_type="dataset",
            local_dir=paths.resolve(cfg.cache_dir).as_posix(),
        )
    )


def _filter_shard(
    con: Any, cfg: FetchConfig, month: str, local: Path, target: Path, remaining: int | None
) -> int:
    """Write the rows of ``local`` that pass the filter into ``target``; return how many."""
    limit = "" if remaining is None else f"\nLIMIT {remaining}"
    con.execute(
        f"""
        COPY (
          SELECT *, {_sql_string(month)} AS month
          FROM read_parquet({_sql_string(local.as_posix())}, hive_partitioning = false)
          WHERE {where_clause(cfg)}{limit}
        ) TO {_sql_string(target.as_posix())} (FORMAT PARQUET)
        """
    )
    row = con.execute(
        f"SELECT count(*) FROM read_parquet({_sql_string(target.as_posix())})"
    ).fetchone()
    return int(row[0]) if row else 0


def _fetch_month(con: Any, cfg: FetchConfig, month: str, target: Path) -> int:
    """Download, filter and delete the month's shards until ``limit`` is reached.

    Parts are kept next to the target until the month is done, so an interrupted month resumes
    from the shard it stopped at instead of starting over.
    """
    parts_dir = target.parent / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    shards = month_shards(cfg, month)
    if not shards:
        raise FileNotFoundError(f"{cfg.dataset} has no parquet shards for {month}")
    kept = 0
    for index, shard in enumerate(shards):
        part = parts_dir / f"part-{index:05d}.parquet"
        if part.is_file() and part.stat().st_size:
            kept += _count(con, part)
            log.info("part %s/%s already filtered (%s games so far)", index + 1, len(shards), kept)
        else:
            remaining = None if cfg.limit is None else cfg.limit - kept
            if remaining is not None and remaining <= 0:
                break
            local = _download(cfg, shard)
            kept += _filter_shard(con, cfg, month, local, part, remaining)
            if not cfg.keep_shards:
                local.unlink(missing_ok=True)
            log.info(
                "shard %s/%s of %s filtered: %s games so far", index + 1, len(shards), month, kept
            )
        if cfg.limit is not None and kept >= cfg.limit:
            log.info("%s reached the limit of %s games", month, cfg.limit)
            break
    parts = sorted(path for path in parts_dir.glob("part-*.parquet") if path.stat().st_size)
    # Counted in rows, not in bytes: a parquet with no rows still has a header and a footer, so
    # an empty month would sail through a size check and land downstream looking like a corpus.
    if not parts or not kept:
        raise RuntimeError(f"no games of {month} passed the filter")
    files = ", ".join(_sql_string(path.as_posix()) for path in parts)
    # `hive_partitioning = false` again: the parts sit under `year=/month=/parts/`, so without it
    # DuckDB would read the directory names as columns and collide with the explicit `month`.
    con.execute(
        f"COPY (SELECT * FROM read_parquet([{files}], hive_partitioning = false, "
        f"union_by_name = true)) TO {_sql_string(target.as_posix())} (FORMAT PARQUET)"
    )
    total = _count(con, target)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return total


def _count(con: Any, path: Path) -> int:
    row = con.execute(
        f"SELECT count(*) FROM read_parquet({_sql_string(path.as_posix())}, "
        "hive_partitioning = false)"
    ).fetchone()
    return int(row[0]) if row else 0


def run(cfg: FetchConfig, dry_run: bool = False, overwrite: bool = False) -> FetchPlan:
    """Plan the fetch and, unless ``dry_run``, execute it month by month.

    With ``dry_run=True`` nothing touches the network or the disk. Otherwise each month is
    materialized as ``<out_dir>/year=YYYY/month=MM/games.parquet`` (so ``limit`` applies per
    month) and a ``manifest.json`` with filters, counts and file hashes is written last. The
    manifest stores file paths relative to ``out_dir`` and ``min_plies`` as
    ``min_plies_deferred`` because plies are only filtered in P1, after UCI conversion.

    A month whose file is already on disk is kept rather than fetched again, unless
    ``overwrite``, and a month that was interrupted resumes from the shard it stopped at.
    """
    fetch_plan = plan(cfg)
    if dry_run:
        return fetch_plan

    out_dir = Path(fetch_plan.out_dir)
    counts: dict[str, int] = {}
    files: list[FileHash] = []
    con = connect(cfg.duckdb)
    try:
        for month, out_path in zip(cfg.months, fetch_plan.out_paths, strict=True):
            target = out_dir / out_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (overwrite or _is_empty(target)):
                log.info("%s is already there, keeping it (pass overwrite=True to refetch)", target)
                counts[month] = _count(con, target)
            else:
                log.info("fetching %s into %s", month, target)
                counts[month] = _fetch_month(con, cfg, month, target)
            files.append(
                FileHash(path=out_path, sha256=_sha256(target), bytes=target.stat().st_size)
            )
    finally:
        con.close()

    filters = cfg.model_dump(exclude={"dataset", "months", "out_dir", "min_plies", "duckdb"})
    filters["min_plies_deferred"] = cfg.min_plies
    manifest = Manifest(
        dataset=cfg.dataset,
        months=list(cfg.months),
        filters=filters,
        counts=counts,
        files=files,
    )
    manifest_path = out_dir / fetch_plan.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return fetch_plan
