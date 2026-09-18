"""Tests for rukh.engine: Stockfish detection and the engine check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rukh import engine
from rukh.cli import app
from rukh.engine import (
    EngineCheckResult,
    EngineError,
    EngineNotFound,
    engine_check,
    find_stockfish,
)

# The two live tests carry only `engine`, so `-m unit` never needs the Stockfish binary.
unit = pytest.mark.unit


@unit
def test_find_stockfish_honours_env(rukh_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = rukh_home / "sf.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("RUKH_STOCKFISH", str(fake))
    assert find_stockfish() == fake


@unit
def test_find_stockfish_env_missing_file_raises(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = rukh_home / "missing.exe"
    monkeypatch.setenv("RUKH_STOCKFISH", str(missing))
    monkeypatch.setattr(engine.shutil, "which", lambda _name: None)
    with pytest.raises(EngineNotFound, match=r"RUKH_STOCKFISH=.*missing\.exe is not a file"):
        find_stockfish()


@unit
def test_find_stockfish_looks_in_tools(rukh_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUKH_STOCKFISH", raising=False)
    monkeypatch.setattr(engine.shutil, "which", lambda _name: None)
    assert find_stockfish() is None
    tools = rukh_home / "tools" / "stockfish"
    tools.mkdir(parents=True)
    binary = tools / "stockfish.exe"
    binary.write_bytes(b"")
    assert find_stockfish() == binary


@unit
def test_find_stockfish_falls_back_to_path(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUKH_STOCKFISH", raising=False)
    monkeypatch.setattr(engine.shutil, "which", lambda _name: str(rukh_home / "on-path"))
    assert find_stockfish() == rukh_home / "on-path"


@unit
def test_engine_check_without_binary_raises(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUKH_STOCKFISH", raising=False)
    monkeypatch.setattr(engine.shutil, "which", lambda _name: None)
    with pytest.raises(EngineNotFound):
        engine_check()


@unit
def test_cli_engine_check_without_binary_fails_cleanly(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUKH_STOCKFISH", raising=False)
    monkeypatch.setattr(engine.shutil, "which", lambda _name: None)
    result = CliRunner().invoke(app, ["engine", "check"])
    assert result.exit_code == 1
    assert "Stockfish" in result.output


class _FakeEngine:
    """Stand-in for ``SimpleEngine``: speaks UCI but is not Stockfish (no ``UCI_Elo``)."""

    def __init__(self) -> None:
        self.id = {"name": "not-stockfish"}
        self.options: dict[str, object] = {}

    def __enter__(self) -> _FakeEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _use_fake_engine(rukh_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binary = rukh_home / "not-stockfish.exe"
    binary.write_bytes(b"")
    monkeypatch.setenv("RUKH_STOCKFISH", str(binary))
    monkeypatch.setattr(
        engine.chess.engine.SimpleEngine, "popen_uci", staticmethod(lambda *_a, **_k: _FakeEngine())
    )
    return binary


@unit
def test_engine_check_without_uci_elo_raises_engine_error(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_engine(rukh_home, monkeypatch)
    with pytest.raises(EngineError, match="does not expose UCI_Elo; is it Stockfish?"):
        engine_check()


@unit
def test_engine_error_is_a_value_error() -> None:
    assert issubclass(EngineError, ValueError)


@unit
def test_cli_engine_check_without_uci_elo_fails_cleanly(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_engine(rukh_home, monkeypatch)
    result = CliRunner().invoke(app, ["engine", "check"])
    assert result.exit_code == 1
    assert "does not expose UCI_Elo" in result.output
    assert "Traceback" not in result.output


@pytest.mark.engine
@pytest.mark.skipif(find_stockfish() is None, reason="Stockfish binary not found")
def test_engine_check_plays_a_short_game() -> None:
    result = engine_check(elo=1400, plies=10, seed=0)
    assert isinstance(result, EngineCheckResult)
    assert result.uci_elo_min == 1320
    assert result.uci_elo_max == 3190
    assert result.elo == 1400
    assert 0 < result.plies_played <= 10
    assert result.name.lower().startswith("stockfish")
    assert result.result in {"1-0", "0-1", "1/2-1/2", "*"}


@pytest.mark.engine
@pytest.mark.skipif(find_stockfish() is None, reason="Stockfish binary not found")
def test_cli_engine_check_json() -> None:
    args = ["engine", "check", "--elo", "1400", "--plies", "6", "--json"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["uci_elo_min"] == 1320
    assert payload["plies_played"] > 0
