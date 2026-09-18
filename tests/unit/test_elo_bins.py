"""Tests for rukh.data.elo_bins: bin formula, per-bin cap and determinism."""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from rukh.data.elo_bins import EloBinsConfig, run

pytestmark = pytest.mark.unit


def _stage(rukh_home: Path, repo_root: Path) -> None:
    src = repo_root / "tests" / "fixtures" / "games.parquet"
    for month in ("01", "02"):
        target = rukh_home / "data" / "uci" / "year=2025" / f"month={month}" / "games.parquet"
        target.parent.mkdir(parents=True)
        shutil.copy(src, target)


def test_run_caps_each_bin(rukh_home: Path, repo_root: Path) -> None:
    _stage(rukh_home, repo_root)
    manifest = run(EloBinsConfig(n_per_bin=2, seed=1))
    frame = pl.read_parquet((rukh_home / "data" / "elo-bins" / "games.parquet").as_posix())
    assert "bin" in frame.columns
    per_bin = frame.group_by("bin").len()
    assert per_bin["len"].max() <= 2
    expected = ((pl.col("white_elo") + pl.col("black_elo")) // 2) // 100 * 100
    assert frame.filter(pl.col("bin") != expected).height == 0
    assert manifest.months == ["2025-01", "2025-02"]
    assert all(v <= 2 for v in manifest.counts.values())
    assert manifest.filters["n_per_bin"] == 2


def test_run_is_deterministic(rukh_home: Path, repo_root: Path) -> None:
    _stage(rukh_home, repo_root)
    run(EloBinsConfig(n_per_bin=1, seed=3))
    first = pl.read_parquet((rukh_home / "data" / "elo-bins" / "games.parquet").as_posix())
    run(EloBinsConfig(n_per_bin=1, seed=3))
    second = pl.read_parquet((rukh_home / "data" / "elo-bins" / "games.parquet").as_posix())
    assert first.equals(second)
    assert first.height == first["bin"].n_unique()
