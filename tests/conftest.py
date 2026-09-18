"""Shared pytest fixtures for the rukh test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root (the folder containing pyproject.toml)."""
    return REPO_ROOT


@pytest.fixture
def rukh_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point RUKH_HOME at a temporary directory so tests never touch the real data/mlruns."""
    monkeypatch.setenv("RUKH_HOME", str(tmp_path))
    return tmp_path
