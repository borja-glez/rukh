"""Tests for rukh.data.db: every connection gets the spill directory, memory limit and threads."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rukh.data.db import TMP_DIRNAME, DuckDbConfig, connect, temp_directory
from rukh.data.pipeline import DUCKDB_STEPS, PipelineConfig

pytestmark = pytest.mark.unit


def _settings(con: object) -> dict[str, object]:
    row = con.execute(  # type: ignore[attr-defined]
        "SELECT current_setting('temp_directory'), current_setting('memory_limit'), "
        "current_setting('threads'), current_setting('preserve_insertion_order')"
    ).fetchone()
    return {
        "temp_directory": row[0],
        "memory_limit": row[1],
        "threads": row[2],
        "preserve_insertion_order": row[3],
    }


def test_connect_applies_the_settings(rukh_home: Path) -> None:
    con = connect(DuckDbConfig(memory_limit="2GB", threads=3))
    try:
        settings = _settings(con)
    finally:
        con.close()
    assert settings["temp_directory"] == (rukh_home / "data" / TMP_DIRNAME).as_posix()
    assert settings["memory_limit"] == "1.8 GiB"  # DuckDB reports GiB
    assert settings["threads"] == 3
    assert (rukh_home / "data" / TMP_DIRNAME).is_dir()
    # Off by default: keeping the input order makes COPY buffer the whole result (D-033).
    assert settings["preserve_insertion_order"] is False


def test_connect_defaults_to_32gb_and_every_core(rukh_home: Path) -> None:
    con = connect()
    try:
        settings = _settings(con)
    finally:
        con.close()
    assert settings["memory_limit"] == "29.8 GiB"  # 32 GB
    assert settings["threads"] == (os.cpu_count() or 4)
    assert temp_directory() == rukh_home / "data" / TMP_DIRNAME


def test_pipeline_shares_the_duckdb_section() -> None:
    cfg = PipelineConfig.model_validate({"duckdb": {"memory_limit": "8GB", "threads": 2}})
    for name in DUCKDB_STEPS:
        step = getattr(cfg, name)
        assert step.duckdb.memory_limit == "8GB" and step.duckdb.threads == 2
    # A step may still override it.
    cfg = PipelineConfig.model_validate(
        {"duckdb": {"memory_limit": "8GB"}, "evals": {"duckdb": {"memory_limit": "1GB"}}}
    )
    assert cfg.evals.duckdb.memory_limit == "1GB"
    assert cfg.positions.duckdb.memory_limit == "8GB"


def test_shipped_pipeline_yaml_declares_duckdb() -> None:
    from rukh.data.pipeline import load_pipeline

    cfg = load_pipeline()
    assert cfg.duckdb.memory_limit == "32GB"
    assert cfg.puzzles.duckdb == cfg.duckdb
