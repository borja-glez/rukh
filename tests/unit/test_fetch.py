"""Tests for rukh.data.fetch: query builder, dry-run plan and local execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from rukh.cli import app
from rukh.config import load_yaml
from rukh.data.fetch import FetchConfig, FetchPlan, build_query, plan, run

pytestmark = pytest.mark.unit

MONTHS = ["2025-01", "2025-02"]

# Columns: Event, WhiteElo, BlackElo, TimeControl, Termination. Exactly one row passes the
# default filters; the '-' time control is what Lichess exports for correspondence games.
FAKE_ROWS = """
    SELECT * FROM (VALUES
      ('Rated Blitz game', 1900, 1850, '300+0', 'Normal'),
      ('Rated Bullet game', 1900, 1850, '60+0', 'Normal'),
      ('Rated Blitz game', 1500, 1850, '300+0', 'Normal'),
      ('Rated Blitz game', 1900, 1850, '300+0', 'Abandoned'),
      ('Rated Atomic variant game', 1900, 1850, '300+0', 'Normal'),
      ('Rated Correspondence game', 1900, 1850, '-', 'Normal')
    ) AS t(Event, WhiteElo, BlackElo, TimeControl, Termination)
"""


def _cfg(**overrides: object) -> FetchConfig:
    return FetchConfig(months=MONTHS, **overrides)  # type: ignore[arg-type]


def _write_hive_source(rukh_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the fake rows under the real ``year=YYYY/month=MM`` layout; point ``_source`` at it."""
    import duckdb

    from rukh.data import fetch as fetch_module

    source_dir = rukh_home / "source" / "year=2025" / "month=01"
    source_dir.mkdir(parents=True)
    duckdb.sql(FAKE_ROWS).write_parquet((source_dir / "part-0.parquet").as_posix())
    monkeypatch.setattr(
        fetch_module, "_source", lambda cfg, month: (source_dir / "*.parquet").as_posix()
    )
    return source_dir


def _read_output(path: Path) -> tuple[list[str], list[tuple[object, ...]]]:
    import duckdb

    rel = duckdb.sql(f"SELECT * FROM read_parquet('{path.as_posix()}', hive_partitioning = false)")
    return list(rel.columns), rel.fetchall()


def test_fetch_config_defaults_match_spec() -> None:
    cfg = _cfg()
    assert cfg.dataset == "Lichess/standard-chess-games"
    assert cfg.min_elo == 1800
    assert cfg.min_base_seconds == 180
    assert cfg.terminations == ["Normal", "Time forfeit"]
    assert cfg.min_plies == 20
    assert cfg.exclude_variants is True
    assert cfg.out_dir == "data/raw"
    assert cfg.limit is None


def test_fetch_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        FetchConfig(months=MONTHS, min_rating=1800)  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", ["2025-1", "2025/01", "202501", "2025-13", ""])
def test_fetch_config_rejects_bad_month(bad: str) -> None:
    with pytest.raises(ValidationError):
        FetchConfig(months=[bad])


def test_fetch_config_requires_at_least_one_month() -> None:
    with pytest.raises(ValidationError):
        FetchConfig(months=[])


@pytest.mark.parametrize(
    "overrides",
    [
        {"terminations": []},
        {"limit": 0},
        {"min_elo": -1},
        {"min_base_seconds": -1},
        {"min_plies": -1},
    ],
    ids=["empty-terminations", "limit-zero", "negative-elo", "negative-base", "negative-plies"],
)
def test_fetch_config_rejects_out_of_range_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        FetchConfig(months=MONTHS, **overrides)  # type: ignore[arg-type]


def test_build_query_has_filters_and_both_months() -> None:
    sql = build_query(_cfg())
    for month in ("month=01", "month=02"):
        assert f"hf://datasets/Lichess/standard-chess-games/data/year=2025/{month}/*.parquet" in sql
    assert sql.count("read_parquet(") == 2
    assert sql.count("hive_partitioning = false") == 2
    assert "UNION ALL" in sql
    assert "WhiteElo >= 1800 AND BlackElo >= 1800" in sql
    assert "TRY_CAST(split_part(TimeControl, '+', 1) AS INTEGER) >= 180" in sql
    assert "Termination IN ('Normal', 'Time forfeit')" in sql
    assert "Event NOT ILIKE '%variant%'" in sql
    assert "LIMIT" not in sql


def test_build_query_optional_parts() -> None:
    sql = build_query(_cfg(exclude_variants=False, limit=1000))
    assert "ILIKE" not in sql
    assert sql.rstrip().endswith("LIMIT 1000")


def test_plan_lists_relative_output_paths_and_manifest(rukh_home: Path) -> None:
    fetch_plan = plan(_cfg())
    assert isinstance(fetch_plan, FetchPlan)
    assert fetch_plan.months == MONTHS
    assert fetch_plan.out_dir == (rukh_home / "data" / "raw").as_posix()
    assert fetch_plan.out_paths == [
        "year=2025/month=01/games.parquet",
        "year=2025/month=02/games.parquet",
    ]
    assert fetch_plan.manifest_path == "manifest.json"
    assert fetch_plan.query == build_query(_cfg())


def test_run_dry_does_not_touch_disk(rukh_home: Path) -> None:
    result = run(_cfg(), dry_run=True)
    assert result == plan(_cfg())
    assert not (rukh_home / "data").exists()


def test_shipped_config_loads(repo_root: Path) -> None:
    cfg = load_yaml(repo_root / "configs" / "data" / "lichess-2025-01-02.yaml", FetchConfig)
    assert cfg.months == MONTHS


def test_cli_data_fetch_dry_run_prints_plan(rukh_home: Path, repo_root: Path) -> None:
    config = repo_root / "configs" / "data" / "lichess-2025-01-02.yaml"
    result = CliRunner().invoke(app, ["data", "fetch", "--config", str(config), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "read_parquet(" in result.output
    assert "2025-01" in result.output and "2025-02" in result.output
    assert (rukh_home / "data" / "raw").as_posix() in result.output
    assert "  year=2025/month=01/games.parquet" in result.output
    assert "manifest.json" in result.output
    assert not (rukh_home / "data").exists()


def test_cli_data_fetch_dry_run_json(rukh_home: Path, repo_root: Path) -> None:
    config = repo_root / "configs" / "data" / "lichess-2025-01-02.yaml"
    result = CliRunner().invoke(
        app, ["data", "fetch", "--config", str(config), "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["months"] == MONTHS
    assert payload["out_paths"] == [
        "year=2025/month=01/games.parquet",
        "year=2025/month=02/games.parquet",
    ]


def test_run_writes_parquet_and_manifest_from_local_source(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rukh.data.manifest import Manifest

    _write_hive_source(rukh_home, monkeypatch)

    cfg = FetchConfig(months=["2025-01"])
    fetch_plan = run(cfg, dry_run=False)

    out_dir = rukh_home / "data" / "raw"
    assert fetch_plan.out_paths == ["year=2025/month=01/games.parquet"]
    out = out_dir / fetch_plan.out_paths[0]
    assert out.is_file()
    columns, rows = _read_output(out)
    assert len(rows) == 1
    assert columns.count("month") == 1
    assert "month_1" not in columns
    assert "year" not in columns

    manifest = Manifest.model_validate_json((out_dir / "manifest.json").read_text("utf-8"))
    assert manifest.counts == {"2025-01": 1}
    assert manifest.files[0].path == "year=2025/month=01/games.parquet"
    assert manifest.filters["min_elo"] == 1800
    assert manifest.filters["min_plies_deferred"] == 20
    assert "min_plies" not in manifest.filters


def test_run_drops_correspondence_games_instead_of_aborting(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_hive_source(rukh_home, monkeypatch)

    fetch_plan = run(FetchConfig(months=["2025-01"], min_elo=0), dry_run=False)

    columns, rows = _read_output(rukh_home / "data" / "raw" / fetch_plan.out_paths[0])
    events = [row[columns.index("Event")] for row in rows]
    assert events == ["Rated Blitz game", "Rated Blitz game"]
