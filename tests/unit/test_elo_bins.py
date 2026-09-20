"""Tests for rukh.data.elo_bins: bin formula, per-bin cap and determinism."""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pydantic
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


def test_run_handles_ids_at_the_edge_of_int64(rukh_home: Path, repo_root: Path) -> None:
    """``game_id`` is a signed 64-bit hash: adding the seed to it would overflow."""
    src = pl.read_parquet((repo_root / "tests" / "fixtures" / "games.parquet").as_posix())
    edges = [2**63 - 1, -(2**63), 2**63 - 2, -(2**63) + 1]
    ids = (edges * ((src.height // len(edges)) + 1))[: src.height]
    target = rukh_home / "data" / "uci" / "year=2025" / "month=01" / "games.parquet"
    target.parent.mkdir(parents=True)
    src.with_columns(pl.Series("game_id", ids, dtype=pl.Int64)).write_parquet(target.as_posix())
    manifest = run(EloBinsConfig(n_per_bin=1, seed=42))
    assert sum(manifest.counts.values()) > 0


def test_run_is_deterministic(rukh_home: Path, repo_root: Path) -> None:
    _stage(rukh_home, repo_root)
    run(EloBinsConfig(n_per_bin=1, seed=3))
    first = pl.read_parquet((rukh_home / "data" / "elo-bins" / "games.parquet").as_posix())
    run(EloBinsConfig(n_per_bin=1, seed=3))
    second = pl.read_parquet((rukh_home / "data" / "elo-bins" / "games.parquet").as_posix())
    assert first.equals(second)
    assert first.height == first["bin"].n_unique()


def _stage_at(rukh_home: Path, repo_root: Path, tree: str, elos: list[int]) -> None:
    """One month of games under ``tree`` whose two ratings are both ``elos[i]``."""
    src = pl.read_parquet((repo_root / "tests" / "fixtures" / "games.parquet").as_posix())
    ratings = (elos * ((src.height // len(elos)) + 1))[: src.height]
    frame = src.with_columns(
        pl.Series("white_elo", ratings, dtype=pl.Int16),
        pl.Series("black_elo", ratings, dtype=pl.Int16),
        pl.Series("game_id", list(range(src.height)), dtype=pl.Int64),
    )
    target = rukh_home / "data" / tree / "year=2025" / "month=01" / "games.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target.as_posix())


def test_two_corpora_become_one_axis(rukh_home: Path, repo_root: Path) -> None:
    """The band below 1800 lives in its own tree and must not be mixed into `data/uci`.

    `data/uci` is what every published model trained on, so the low slice is fetched apart and
    only joins it here, where the sample that teaches the Elo header is built.
    """
    _stage_at(rukh_home, repo_root, "uci", [2000, 2400])
    _stage_at(rukh_home, repo_root, "uci-low", [1200, 1600])
    manifest = run(EloBinsConfig(sources=["data/uci", "data/uci-low"], n_per_bin=5, bin_width=400))
    assert sorted(int(b) for b in manifest.counts) == [1200, 1600, 2000, 2400]
    assert manifest.filters["sources"] == ["data/uci", "data/uci-low"]


def test_the_sample_is_flat_over_bands_however_skewed_the_source_is(
    rukh_home: Path, repo_root: Path
) -> None:
    """Flatness is the whole point: it teaches what a header means, not how common it is."""
    src = pl.read_parquet((repo_root / "tests" / "fixtures" / "games.parquet").as_posix())
    # 15 games at 1200 against 5 at 2000, the shape a real corpus has.
    ratings = [1200] * 15 + [2000] * 5
    frame = src.with_columns(
        pl.Series("white_elo", ratings[: src.height], dtype=pl.Int16),
        pl.Series("black_elo", ratings[: src.height], dtype=pl.Int16),
        pl.Series("game_id", list(range(src.height)), dtype=pl.Int64),
    )
    target = rukh_home / "data" / "uci" / "year=2025" / "month=01" / "games.parquet"
    target.parent.mkdir(parents=True)
    frame.write_parquet(target.as_posix())
    manifest = run(EloBinsConfig(n_per_bin=5, bin_width=400))
    assert manifest.counts == {"1200": 5, "2000": 5}


def test_bands_outside_the_range_are_dropped(rukh_home: Path, repo_root: Path) -> None:
    _stage_at(rukh_home, repo_root, "uci", [800, 1200, 2000, 3000])
    manifest = run(EloBinsConfig(n_per_bin=5, bin_width=400, min_bin=1200, max_bin=2000))
    assert sorted(int(b) for b in manifest.counts) == [1200, 2000]
    assert manifest.filters["min_bin"] == 1200


def test_the_band_width_is_in_the_bin_formula_the_manifest_publishes(
    rukh_home: Path, repo_root: Path
) -> None:
    _stage_at(rukh_home, repo_root, "uci", [1000, 1100])
    wide = run(EloBinsConfig(n_per_bin=50, bin_width=200))
    assert "// 200 * 200" in str(wide.filters["bin"])
    assert sorted(int(b) for b in wide.counts) == [1000]  # both ratings fall in one 200-wide band
    narrow = run(EloBinsConfig(n_per_bin=50, bin_width=100))
    assert sorted(int(b) for b in narrow.counts) == [1000, 1100]


def test_an_empty_source_list_is_refused() -> None:
    with pytest.raises(pydantic.ValidationError):
        EloBinsConfig(sources=[])


def test_an_inverted_band_range_is_refused() -> None:
    with pytest.raises(pydantic.ValidationError):
        EloBinsConfig(min_bin=2000, max_bin=1200)


def test_a_missing_source_says_which_one(rukh_home: Path, repo_root: Path) -> None:
    _stage_at(rukh_home, repo_root, "uci", [1500])
    with pytest.raises(FileNotFoundError, match="uci-low"):
        run(EloBinsConfig(sources=["data/uci", "data/uci-low"]))
