"""Tests for rukh.data.positions: fen4 normalization, phase labels and deduplication."""

from __future__ import annotations

import shutil
from pathlib import Path

import chess
import polars as pl
import pytest
from typer.testing import CliRunner

from rukh.cli import app
from rukh.data.positions import PositionsConfig, fen4, phase, run, walk_game

pytestmark = pytest.mark.unit


def test_fen4_drops_counters_and_keeps_castling_and_ep() -> None:
    board = chess.Board()
    for move in ("e2e4", "a7a6", "e4e5", "d7d5"):
        board.push_uci(move)
    assert board.fen().split()[4:] == ["0", "3"]
    assert fen4(board) == "rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6"
    board.push_uci("e5d6")
    assert fen4(board).endswith(" b KQkq -")


def test_phase_thresholds() -> None:
    assert phase(1, 32) == "opening"
    assert phase(10, 32) == "opening"
    assert phase(11, 32) == "middlegame"
    assert phase(11, 14) == "middlegame"
    assert phase(11, 13) == "endgame"
    assert phase(80, 5) == "endgame"


def test_walk_game_rows() -> None:
    rows = list(walk_game(7, "e2e4 e7e5 g1f3", "1-0"))
    assert len(rows) == 3
    fen, game_id, ply, last_move, result, ph = rows[0]
    assert fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    assert (game_id, ply, last_move, result, ph) == (7, 1, "e2e4", "1-0", "opening")
    assert rows[2][2] == 3 and rows[2][3] == "g1f3"


def _stage_uci(rukh_home: Path, repo_root: Path) -> None:
    target = rukh_home / "data" / "uci" / "year=2025" / "month=01" / "games.parquet"
    target.parent.mkdir(parents=True)
    shutil.copy(repo_root / "tests" / "fixtures" / "games.parquet", target)


def test_run_dedupes_with_counts(rukh_home: Path, repo_root: Path) -> None:
    _stage_uci(rukh_home, repo_root)
    manifest = run(PositionsConfig(workers=1))
    out = rukh_home / "data" / "positions" / "positions.parquet"
    frame = pl.read_parquet(out.as_posix())
    assert frame.columns == ["fen4", "game_id", "ply", "last_move", "result", "phase", "n_seen"]
    total = int(
        pl.read_parquet((repo_root / "tests" / "fixtures" / "games.parquet").as_posix())[
            "n_plies"
        ].sum()
    )
    assert manifest.counts["positions"] == total
    assert manifest.counts["distinct"] == frame.height < total
    assert frame["fen4"].n_unique() == frame.height
    after_e4 = frame.filter(
        pl.col("fen4") == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    )
    assert after_e4.height == 1 and int(after_e4["n_seen"][0]) > 1
    assert after_e4["ply"][0] == 1 and after_e4["last_move"][0] == "e2e4"
    assert set(frame["phase"].unique().to_list()) == {"opening", "middlegame", "endgame"}
    assert manifest.files[0].path == "positions.parquet"


def test_run_with_two_workers(rukh_home: Path, repo_root: Path) -> None:
    """The spawn pool path, exercised end to end on the fixture month."""
    _stage_uci(rukh_home, repo_root)
    manifest = run(PositionsConfig(workers=2, n_games=4))
    frame = pl.read_parquet((rukh_home / "data" / "positions" / "positions.parquet").as_posix())
    assert manifest.counts["distinct"] == frame.height > 0
    assert frame["fen4"].n_unique() == frame.height


def test_run_respects_caps(rukh_home: Path, repo_root: Path) -> None:
    _stage_uci(rukh_home, repo_root)
    manifest = run(PositionsConfig(workers=1, n_games=2, max_positions=30))
    assert manifest.counts["distinct"] == 30
    assert manifest.counts["positions"] == 33 + 7


def test_cli_positions(rukh_home: Path, repo_root: Path) -> None:
    _stage_uci(rukh_home, repo_root)
    cfg = rukh_home / "pipeline.yaml"
    cfg.write_text("positions:\n  workers: 1\n  n_games: 3\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["data", "positions", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "distinct:" in result.output
