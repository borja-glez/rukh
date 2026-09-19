"""Shared pytest fixtures for the rukh test suite."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

ANSI = re.compile(r"\[[0-9;]*m")
"""Rich styles the help output; the escapes break naive substring assertions."""


def cli_options(*path: str) -> set[str]:
    """Every flag a CLI command declares, read from the parser instead of from ``--help``.

    Asserting ``"--port" in result.output`` looks harmless and is not: Typer renders the help
    through Rich, and when colour is enabled — which it is on the GitHub runner, and is not on a
    plain Windows shell — the style escapes land *inside* the option name, so the substring is
    missing from output that visibly contains the flag. Five tests passed locally and failed in CI
    for exactly that reason. The parser is what actually defines the interface, so ask it.
    """
    import typer.main

    from rukh.cli import app

    command = typer.main.get_command(app)
    for name in path:
        command = command.commands[name]  # type: ignore[attr-defined]
    return {opt for param in command.params for opt in param.opts}


def plain(text: str) -> str:
    """``text`` without the ANSI escapes Rich adds when it thinks the output is a terminal."""
    return ANSI.sub("", text)


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root (the folder containing pyproject.toml)."""
    return REPO_ROOT


@pytest.fixture
def rukh_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point RUKH_HOME at a temporary directory so tests never touch the real data/mlruns."""
    monkeypatch.setenv("RUKH_HOME", str(tmp_path))
    return tmp_path
