"""Fetch a filtered slice of ``Lichess/standard-chess-games`` with DuckDB over ``hf://``.

Only the games that pass the filters are materialized locally (predicate pushdown on the
remote parquet files). The ply count (``min_plies``) is not applied here: it is enforced in P1
once ``movetext`` has been converted to UCI with python-chess, because counting plies from SAN
with comments (``%clk``, ``%eval``) in SQL is unreliable. The manifest therefore records it as
``min_plies_deferred``. Variant exclusion by ``Event`` is a coarse first pass that P1 refines.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rukh import paths
from rukh.config import BaseConfig
from rukh.data.db import DuckDbConfig, connect
from rukh.data.manifest import FileHash, Manifest

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


def build_query(cfg: FetchConfig) -> str:
    """Build the DuckDB SQL that selects the filtered games, one ``read_parquet`` per month.

    ``TRY_CAST`` drops rows whose base time is not an integer (correspondence games export
    ``TimeControl = '-'``). ``hive_partitioning = false`` stops DuckDB from injecting ``year``
    and ``month`` columns from the path, which would collide with the explicit ``month`` alias.
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
    where = "\n    AND ".join(conditions)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cfg: FetchConfig, dry_run: bool = False) -> FetchPlan:
    """Plan the fetch and, unless ``dry_run``, execute it month by month.

    With ``dry_run=True`` nothing touches the network or the disk. Otherwise each month is
    materialized as ``<out_dir>/year=YYYY/month=MM/games.parquet`` (so ``limit`` applies per
    month) and a ``manifest.json`` with filters, counts and file hashes is written last. The
    manifest stores file paths relative to ``out_dir`` and ``min_plies`` as
    ``min_plies_deferred`` because plies are only filtered in P1, after UCI conversion.
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
            month_query = build_query(cfg.model_copy(update={"months": [month]}))
            target_sql = _sql_string(target.as_posix())
            con.execute(f"COPY ({month_query}) TO {target_sql} (FORMAT PARQUET)")
            row = con.execute(
                f"SELECT count(*) FROM read_parquet({target_sql}, hive_partitioning = false)"
            ).fetchone()
            counts[month] = int(row[0]) if row else 0
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
