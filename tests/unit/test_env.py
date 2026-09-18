"""Tests for rukh.env and the `rukh info` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rukh import paths
from rukh.cli import app
from rukh.env import EnvReport, collect

pytestmark = pytest.mark.unit


def test_collect_returns_report_with_python_and_mlflow(rukh_home: Path) -> None:
    report = collect()
    assert isinstance(report, EnvReport)
    assert report.python.startswith("3.12")
    assert report.mlflow_uri.startswith("sqlite:///")
    assert report.data_dir == str(rukh_home / "data")
    assert report.mlruns_dir == str(rukh_home / "mlruns")
    assert isinstance(report.cuda, bool)


def test_collect_survives_missing_torch(rukh_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    report = collect()
    assert report.torch is None
    assert report.cuda is False
    assert report.gpu is None


def test_paths_follow_rukh_home(rukh_home: Path) -> None:
    assert paths.root() == rukh_home
    assert paths.data_dir() == rukh_home / "data"
    assert paths.mlruns_dir() == rukh_home / "mlruns"
    assert paths.tools_dir() == rukh_home / "tools"


def test_paths_default_to_repo_root(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.delenv("RUKH_HOME", raising=False)
    assert paths.root() == repo_root


def test_info_json_is_parseable(rukh_home: Path) -> None:
    result = CliRunner().invoke(app, ["info", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(EnvReport.model_fields) <= set(payload)
    assert payload["python"].startswith("3.12")
    assert payload["mlflow_uri"].startswith("sqlite:///")


def test_info_human_output_mentions_python(rukh_home: Path) -> None:
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert "python" in result.output
    assert "3.12" in result.output
