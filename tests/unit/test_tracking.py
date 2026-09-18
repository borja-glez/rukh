"""Tests for rukh.tracking: local MLflow store and run helper."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from rukh import __version__
from rukh.tracking import flatten, start_run, tracking_uri

pytestmark = pytest.mark.unit


def test_tracking_uri_points_to_sqlite_under_mlruns(rukh_home: Path) -> None:
    uri = tracking_uri(create=False)
    assert uri == f"sqlite:///{(rukh_home / 'mlruns' / 'mlflow.db').as_posix()}"
    assert not (rukh_home / "mlruns").exists()
    tracking_uri()
    assert (rukh_home / "mlruns").is_dir()


def test_flatten_nested_dict() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": "x"}}, "e": [1, 2]}) == {
        "a.b": "1",
        "a.c.d": "x",
        "e": "[1, 2]",
    }


def test_start_run_logs_flattened_params_and_tags(rukh_home: Path) -> None:
    with start_run("t", {"a": {"b": 1}}, tags={"purpose": "test"}) as run:
        run_id = run.info.run_id
    assert (rukh_home / "mlruns" / "mlflow.db").is_file()

    mlflow.set_tracking_uri(tracking_uri())
    runs = mlflow.search_runs(experiment_names=["rukh"], output_format="list")
    found = [r for r in runs if r.info.run_id == run_id]
    assert len(found) == 1
    logged = found[0]
    assert logged.info.run_name == "t"
    assert logged.data.params["a.b"] == "1"
    assert logged.data.tags["rukh_version"] == __version__
    assert logged.data.tags["purpose"] == "test"
    assert logged.info.status == "FINISHED"


def test_cli_has_mlflow_ui_command() -> None:
    from typer.testing import CliRunner

    from rukh.cli import app

    result = CliRunner().invoke(app, ["mlflow", "ui", "--help"])
    assert result.exit_code == 0, result.output
    assert "--port" in result.output
