"""Tests for rukh.eval.suite and the ``rukh eval`` command: missing inputs become notes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from typer.testing import CliRunner

from rukh.cli import app
from rukh.eval.suite import EvalConfig, config_path, evaluate, load_suite, run_suite
from rukh.models import DecoderConfig, MoveDecoder
from rukh.train import save_checkpoint

pytestmark = pytest.mark.unit

TOY = DecoderConfig(vocab_size=2030, n_layer=2, n_head=2, d_model=32, block=64)
GAME = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6"


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = MoveDecoder(TOY)
    path = tmp_path / "tiny" / "best.pt"
    return save_checkpoint(
        path, step=3, model=model, optimizer=None, cfg={}, model_cfg=TOY.model_dump()
    )


def quick_config(rukh_home: Path, games: Path | None = None) -> EvalConfig:
    """A suite that touches no engine (``elo_games=0``) and no MLflow store."""
    return EvalConfig(
        games=str(games) if games else str(rukh_home / "missing.parquet"),
        puzzles=str(rukh_home / "missing-puzzles.parquet"),
        legality_positions=2,
        accuracy_positions=2,
        elo_games=0,
        track=False,
        out_dir=str(rukh_home / "artifacts" / "eval"),
        web_results=str(rukh_home / "artifacts" / "web" / "results.json"),
        cache_db=str(rukh_home / "artifacts" / "eval" / "cache.sqlite"),
    )


@pytest.fixture
def games(tmp_path: Path) -> Path:
    import polars as pl

    path = tmp_path / "games.parquet"
    pl.DataFrame(
        {
            "game_id": [1, 2],
            "uci": [GAME, GAME],
            "white_elo": [1850, 2150],
            "black_elo": [1950, 2050],
        }
    ).write_parquet(path)
    return path


def test_both_shipped_suites_load(repo_root: Path) -> None:
    for suite in ("full", "quick"):
        assert config_path(suite).is_file()
        assert load_suite(suite).legality_positions > 0
    assert load_suite("quick").legality_positions < load_suite("full").legality_positions


def test_an_unknown_suite_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown suite"):
        config_path("enormous")


def test_missing_inputs_become_notes_not_failures(rukh_home: Path, checkpoint: Path) -> None:
    result = evaluate(checkpoint, quick_config(rukh_home), suite="quick")
    assert result.stage == "tiny"
    assert result.params == MoveDecoder(TOY).num_params(non_embedding=False)
    assert (result.legality, result.accuracy, result.puzzles, result.elo) == (None,) * 4
    assert any("validation games not found" in note for note in result.notes)
    assert any("puzzles not found" in note for note in result.notes)
    assert any("no games were played" in note for note in result.notes)


def test_a_suite_with_games_measures_legality_and_accuracy(
    rukh_home: Path, checkpoint: Path, games: Path
) -> None:
    result, report = run_suite(checkpoint, quick_config(rukh_home, games), suite="quick")
    assert result.legality is not None and 0.0 <= result.legality.rate <= 1.0
    assert result.accuracy is not None and result.accuracy.positions == 2
    assert Path(report.markdown).is_file()
    table = json.loads(Path(report.web or "").read_text(encoding="utf-8"))
    assert [row["stage"] for row in table["rows"]] == ["tiny"]


def test_the_cli_writes_a_report(rukh_home: Path, checkpoint: Path, games: Path) -> None:
    config = rukh_home / "suite.yaml"
    config.write_text(
        yaml.safe_dump(quick_config(rukh_home, games).model_dump(mode="json")), encoding="utf-8"
    )
    result = CliRunner().invoke(
        app,
        ["eval", "--model", str(checkpoint), "--suite", "quick", "--config", str(config)],
    )
    assert result.exit_code == 0, result.output
    assert "legality:" in result.output
    assert "report:" in result.output


def test_the_cli_rejects_an_unknown_suite(rukh_home: Path, checkpoint: Path) -> None:
    result = CliRunner().invoke(app, ["eval", "--model", str(checkpoint), "--suite", "enormous"])
    assert result.exit_code == 2
