"""Tests for rukh.data.puzzles: filters, bands and the deterministic disjoint split."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rukh.data import puzzles as puzzles_module
from rukh.data.puzzles import PuzzlesConfig, band, run, split_frame, split_key

pytestmark = pytest.mark.unit


def test_band_edges() -> None:
    assert band(999) is None
    assert band(1000) == "1000-1500"
    assert band(1499) == "1000-1500"
    assert band(1500) == "1500-2000"
    assert band(2000) == "2000+"
    assert band(3100) == "2000+"


def test_split_key_is_stable() -> None:
    assert split_key("abc12", 42) == split_key("abc12", 42)
    assert split_key("abc12", 42) != split_key("abc12", 43)
    assert split_key("abc12", 42) != split_key("abc13", 42)


def _remote(path: Path) -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "PuzzleId": f"p{i:05d}",
                "GameId": f"g{i}",
                "FEN": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "Moves": "f1b5 a7a6",
                "Rating": 900 + i * 40,
                "RatingDeviation": 150 if i % 10 == 0 else 80,
                "Popularity": 90,
                "NbPlays": 50 if i % 7 == 0 else 500,
                "Themes": ["fork", "short"],
                "OpeningTags": ["Ruy_Lopez"],
            }
        )
    pl.DataFrame(rows).write_parquet(path.as_posix())


def test_run_filters_bands_and_splits(rukh_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    remote = rukh_home / "remote.parquet"
    _remote(remote)
    monkeypatch.setattr(puzzles_module, "_source", lambda cfg: remote.as_posix())
    cfg = PuzzlesConfig(n_test=3, n_train=5)
    manifest = run(cfg)
    frame = pl.read_parquet((rukh_home / "data" / "puzzles" / "puzzles.parquet").as_posix())
    assert frame.columns == ["puzzle_id", "fen", "moves", "rating", "themes", "band", "split"]
    assert frame["rating"].min() >= 1000
    assert frame["puzzle_id"].n_unique() == frame.height
    for b in ("1000-1500", "1500-2000", "2000+"):
        test = frame.filter((pl.col("band") == b) & (pl.col("split") == "test"))
        train = frame.filter((pl.col("band") == b) & (pl.col("split") == "train"))
        assert test.height == 3 and 0 < train.height <= 5
        assert not set(test["puzzle_id"]) & set(train["puzzle_id"])
    assert manifest.counts["2000+/test"] == 3
    # Rows dropped by quality filters never appear.
    ids = set(frame["puzzle_id"].to_list())
    assert "p00000" not in ids and "p00007" not in ids and "p00010" not in ids


def test_split_is_deterministic_and_seed_sensitive(rukh_home: Path) -> None:
    remote = rukh_home / "remote.parquet"
    _remote(remote)
    frame = (
        pl.read_parquet(remote.as_posix())
        .rename(
            {
                "PuzzleId": "puzzle_id",
                "FEN": "fen",
                "Moves": "moves",
                "Rating": "rating",
                "Themes": "themes",
            }
        )
        .select("puzzle_id", "fen", "moves", "rating", "themes")
    )
    cfg = PuzzlesConfig(n_test=4, n_train=6, seed=42)
    a = split_frame(frame, cfg)
    b = split_frame(frame, cfg)
    assert a.equals(b)
    c = split_frame(frame, cfg.model_copy(update={"seed": 7}))
    assert set(a.filter(pl.col("split") == "test")["puzzle_id"]) != set(
        c.filter(pl.col("split") == "test")["puzzle_id"]
    )
