"""One configured DuckDB connection for the whole pipeline.

Every step joins or groups files far larger than RAM, so a default connection is a foot-gun:
DuckDB spills to the system temp directory (often a small system drive) and, without a memory
limit, competes with the worker processes. ``connect`` sets ``temp_directory`` to
``<data>/duckdb-tmp`` (created on demand, gitignored), ``memory_limit`` and ``threads`` on
every connection.

It also turns off ``preserve_insertion_order``. With it on (the default) a ``COPY`` of a large
streaming scan buffers the whole result so the output rows keep the input order; the first real
fetch of a month grew to 20 GB of resident memory for a 2.5 GB file. Nothing downstream cares
about the order of the games, so the buffering buys nothing and costs the machine.

Finally, it signs the requests to ``hf://``. Reading a month of Lichess means a few hundred range
requests against the same host, and anonymous ones get ``HTTP 429 Too Many Requests`` partway
through -- which arrives as a failed ``COPY`` after minutes of work, with no partial file to
resume from. A token raises the allowance and identifies the traffic, and the retry budget is
raised from three attempts to eight because a 429 is a "wait", not a "no".
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from rukh import paths
from rukh.config import BaseConfig

if TYPE_CHECKING:
    import duckdb

log = logging.getLogger(__name__)

TMP_DIRNAME = "duckdb-tmp"


class DuckDbConfig(BaseConfig):
    """Settings applied to every DuckDB connection (``duckdb:`` in ``pipeline.yaml``)."""

    memory_limit: str = "32GB"
    threads: int = Field(default=0, ge=0)
    """``0`` = ``cpu_count()``."""
    preserve_insertion_order: bool = False
    """Keep the input order in the output. Off: it buffers the whole result of a ``COPY``."""
    http_retries: int = Field(default=8, ge=0)
    """Attempts per HTTP request. The default of 3 is not enough for a rate-limited host."""
    http_retry_wait_ms: int = Field(default=500, ge=0)
    """First backoff, doubled by ``http_retry_backoff`` on each attempt."""
    http_timeout: int = Field(default=120, ge=1)
    """Seconds before a single request is given up on; the default 30 is tight over ``hf://``."""
    authenticate: bool = True
    """Sign ``hf://`` requests with the locally stored Hugging Face token when there is one."""


def temp_directory() -> Path:
    """``<data_dir>/duckdb-tmp``, created if missing: spills stay on the data drive."""
    tmp = paths.data_dir() / TMP_DIRNAME
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def hf_token() -> str | None:
    """The Hugging Face token this machine is logged in with, or None.

    Read through ``huggingface_hub`` rather than from the environment so it picks up the token
    that ``huggingface-cli login`` wrote, which is where it usually lives on a workstation.
    """
    try:
        from huggingface_hub import get_token
    except ImportError:  # pragma: no cover - the hub is a hard dependency
        return None
    token = get_token()
    return token or None


def authenticate(con: duckdb.DuckDBPyConnection, token: str | None = None) -> bool:
    """Give the connection a Hugging Face secret; return whether one was installed."""
    secret = token if token is not None else hf_token()
    if not secret:
        log.info("no Hugging Face token found: hf:// reads go out anonymous and rate-limited")
        return False
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN ?)", [secret])
    return True


def connect(cfg: DuckDbConfig | None = None) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection with the spill directory, memory limit and threads set."""
    import duckdb

    settings = cfg or DuckDbConfig()
    con = duckdb.connect()
    con.execute("SET temp_directory = ?", [temp_directory().as_posix()])
    con.execute("SET memory_limit = ?", [settings.memory_limit])
    con.execute("SET threads = ?", [settings.threads or (os.cpu_count() or 4)])
    con.execute("SET preserve_insertion_order = ?", [settings.preserve_insertion_order])
    con.execute("SET http_retries = ?", [settings.http_retries])
    con.execute("SET http_retry_wait_ms = ?", [settings.http_retry_wait_ms])
    con.execute("SET http_timeout = ?", [settings.http_timeout])
    if settings.authenticate:
        authenticate(con)
    return con
