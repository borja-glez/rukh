"""One configured DuckDB connection for the whole pipeline.

Every step joins or groups files far larger than RAM, so a default connection is a foot-gun:
DuckDB spills to the system temp directory (often a small system drive) and, without a memory
limit, competes with the worker processes. ``connect`` sets ``temp_directory`` to
``<data>/duckdb-tmp`` (created on demand, gitignored), ``memory_limit`` and ``threads`` on
every connection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from rukh import paths
from rukh.config import BaseConfig

if TYPE_CHECKING:
    import duckdb

TMP_DIRNAME = "duckdb-tmp"


class DuckDbConfig(BaseConfig):
    """Settings applied to every DuckDB connection (``duckdb:`` in ``pipeline.yaml``)."""

    memory_limit: str = "32GB"
    threads: int = Field(default=0, ge=0)
    """``0`` = ``cpu_count()``."""


def temp_directory() -> Path:
    """``<data_dir>/duckdb-tmp``, created if missing: spills stay on the data drive."""
    tmp = paths.data_dir() / TMP_DIRNAME
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def connect(cfg: DuckDbConfig | None = None) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection with the spill directory, memory limit and threads set."""
    import duckdb

    settings = cfg or DuckDbConfig()
    con = duckdb.connect()
    con.execute("SET temp_directory = ?", [temp_directory().as_posix()])
    con.execute("SET memory_limit = ?", [settings.memory_limit])
    con.execute("SET threads = ?", [settings.threads or (os.cpu_count() or 4)])
    return con
