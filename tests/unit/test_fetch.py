"""Tests for rukh.data.fetch: query builder, dry-run plan and local execution."""

from __future__ import annotations

import json
import shutil
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


def _write_hive_source(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch, shards: int = 1
) -> list[Path]:
    """Stand in for the Hub: ``shards`` local parquets, and no network anywhere in the run.

    The fetch downloads a shard at a time now, so the two seams a test has to hold are the
    listing (which shards exist) and the download (where one landed). Nothing else is faked: the
    filter, the part files and the merge are the real ones.
    """
    import duckdb

    from rukh.data import fetch as fetch_module

    source_dir = rukh_home / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(shards):
        path = source_dir / f"shard-{index}.parquet"
        duckdb.sql(FAKE_ROWS).write_parquet(path.as_posix())
        written.append(path)
    names = [f"data/year=2025/month=01/train-{i:05d}.parquet" for i in range(shards)]
    downloads = rukh_home / "downloads"
    downloads.mkdir(exist_ok=True)

    def fake_download(cfg: object, shard: str) -> Path:
        """A download is a *copy*: the run deletes what it downloaded, and must not eat the Hub."""
        target = downloads / Path(shard).name
        shutil.copy(written[names.index(shard)], target)
        return target

    monkeypatch.setattr(fetch_module, "month_shards", lambda cfg, month: list(names))
    monkeypatch.setattr(fetch_module, "_download", fake_download)
    return written


def _read_output(path: Path) -> tuple[list[str], list[tuple[object, ...]]]:
    import duckdb

    rel = duckdb.sql(f"SELECT * FROM read_parquet('{path.as_posix()}', hive_partitioning = false)")
    return list(rel.columns), rel.fetchall()


def test_fetch_config_defaults_match_spec() -> None:
    cfg = _cfg()
    assert cfg.dataset == "Lichess/standard-chess-games"
    assert cfg.min_elo == 1800
    assert cfg.max_elo is None
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
        {"max_elo": -1},
        {"min_base_seconds": -1},
        {"min_plies": -1},
    ],
    ids=[
        "empty-terminations",
        "limit-zero",
        "negative-elo",
        "negative-max-elo",
        "negative-base",
        "negative-plies",
    ],
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


def test_build_query_bounds_both_ratings_when_max_elo_is_set() -> None:
    """The band a second corpus fills is closed at both ends, and on both players.

    The project's corpus stops at 1800 from below and has no ceiling, so the Elo tokens under
    `<w1800>` never trained. Filling that band means fetching games where *both* players are in
    it: a 1200 paired against a 2400 is a mismatch, not a 1200-rated game.
    """
    sql = build_query(_cfg(min_elo=1000, max_elo=1799))
    assert "WhiteElo >= 1000 AND BlackElo >= 1000" in sql
    assert "WhiteElo <= 1799 AND BlackElo <= 1799" in sql


def test_build_query_omits_the_ceiling_when_there_is_none() -> None:
    assert "<=" not in build_query(_cfg())


def test_fetch_config_rejects_a_ceiling_below_the_floor() -> None:
    with pytest.raises(ValidationError):
        FetchConfig(months=MONTHS, min_elo=1800, max_elo=1500)


def test_shipped_low_band_config_closes_the_gap_left_by_the_first_corpus(repo_root: Path) -> None:
    """The two shipped corpora must tile the Elo axis without overlapping or leaving a hole."""
    high = load_yaml(repo_root / "configs" / "data" / "lichess-2025-01-02.yaml", FetchConfig)
    low = load_yaml(repo_root / "configs" / "data" / "lichess-low.yaml", FetchConfig)
    assert low.max_elo is not None
    assert low.max_elo + 1 == high.min_elo
    assert low.months == high.months
    assert (low.min_base_seconds, low.terminations, low.min_plies) == (
        high.min_base_seconds,
        high.terminations,
        high.min_plies,
    )


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


def test_a_month_already_on_disk_is_not_fetched_again(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading a month is tens of minutes of range requests and the host answers 429 eventually.

    Without this, the second month failing would throw away the first one, which is the
    expensive half of the work and is already sitting correct on disk.
    """
    _write_hive_source(rukh_home, monkeypatch)
    cfg = FetchConfig(months=["2025-01"])
    run(cfg)
    target = rukh_home / "data" / "raw" / "year=2025" / "month=01" / "games.parquet"
    stamp = target.stat().st_mtime_ns

    run(cfg)
    assert target.stat().st_mtime_ns == stamp  # kept, not rewritten
    counts = json.loads((rukh_home / "data" / "raw" / "manifest.json").read_text())["counts"]
    assert counts == {"2025-01": 1}  # and it still gets counted into the manifest

    run(cfg, overwrite=True)
    assert target.stat().st_mtime_ns != stamp


def test_the_stub_a_failed_copy_leaves_behind_is_refetched(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-byte file is what a `COPY` that died mid-flight leaves; it is not a fetched month."""
    _write_hive_source(rukh_home, monkeypatch)
    target = rukh_home / "data" / "raw" / "year=2025" / "month=01" / "games.parquet"
    target.parent.mkdir(parents=True)
    target.touch()
    run(FetchConfig(months=["2025-01"]))
    assert target.stat().st_size > 0


def test_run_drops_correspondence_games_instead_of_aborting(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_hive_source(rukh_home, monkeypatch)

    fetch_plan = run(FetchConfig(months=["2025-01"], min_elo=0), dry_run=False)

    columns, rows = _read_output(rukh_home / "data" / "raw" / fetch_plan.out_paths[0])
    events = [row[columns.index("Event")] for row in rows]
    assert events == ["Rated Blitz game", "Rated Blitz game"]


def test_the_limit_stops_the_download_not_only_the_result(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For a slice of the rating range this is the difference between 8 shards and 72.

    Each shard is about a gigabyte, so a limit that only trimmed the answer would still pay for
    the whole month in bandwidth and in time.
    """
    from rukh.data import fetch as fetch_module

    _write_hive_source(rukh_home, monkeypatch, shards=5)
    asked: list[str] = []
    real = fetch_module._download
    monkeypatch.setattr(
        fetch_module,
        "_download",
        lambda cfg, shard: (asked.append(shard), real(cfg, shard))[1],
    )
    run(FetchConfig(months=["2025-01"], limit=2))
    assert len(asked) == 2  # one row survives the filter per shard; two shards is enough
    counts = json.loads((rukh_home / "data" / "raw" / "manifest.json").read_text())["counts"]
    assert counts == {"2025-01": 2}


def test_an_interrupted_month_resumes_at_the_shard_it_stopped_on(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rukh.data import fetch as fetch_module

    _write_hive_source(rukh_home, monkeypatch, shards=3)
    month_dir = rukh_home / "data" / "raw" / "year=2025" / "month=01"
    parts = month_dir / "parts"
    parts.mkdir(parents=True)
    # What a run that died after its first shard leaves behind.
    shutil.copy(rukh_home / "source" / "shard-0.parquet", parts / "part-00000.parquet")

    asked: list[str] = []
    real = fetch_module._download
    monkeypatch.setattr(
        fetch_module,
        "_download",
        lambda cfg, shard: (asked.append(shard), real(cfg, shard))[1],
    )
    run(FetchConfig(months=["2025-01"]))
    assert [Path(name).name for name in asked] == ["train-00001.parquet", "train-00002.parquet"]
    assert not parts.exists()  # the parts are cleaned up once the month is merged


def test_a_downloaded_shard_is_deleted_once_it_has_been_filtered(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A month is 72 GB of shards and the filtered result is a few. Peak disk is one shard."""
    _write_hive_source(rukh_home, monkeypatch, shards=3)
    run(FetchConfig(months=["2025-01"]))
    assert not list((rukh_home / "downloads").glob("*.parquet"))


def test_keep_shards_leaves_them_for_a_second_pass(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_hive_source(rukh_home, monkeypatch, shards=2)
    run(FetchConfig(months=["2025-01"], keep_shards=True))
    assert len(list((rukh_home / "downloads").glob("*.parquet"))) == 2


def test_a_month_the_dataset_does_not_have_says_so(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rukh.data import fetch as fetch_module

    monkeypatch.setattr(fetch_module, "month_shards", lambda cfg, month: [])
    with pytest.raises(FileNotFoundError, match="no parquet shards"):
        run(FetchConfig(months=["2025-01"]))


def test_a_filter_that_keeps_nothing_is_an_error_not_an_empty_file(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty games.parquet downstream would look like a corpus, and it is a failed fetch."""
    _write_hive_source(rukh_home, monkeypatch)
    with pytest.raises(RuntimeError, match="passed the filter"):
        run(FetchConfig(months=["2025-01"], min_elo=3000))
