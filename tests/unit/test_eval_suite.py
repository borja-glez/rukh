"""Tests for rukh.eval.suite and the ``rukh eval`` command: missing inputs become notes."""

from __future__ import annotations

import json
import string
from pathlib import Path

import pytest
import torch
import yaml
from typer.testing import CliRunner

from rukh.cli import app
from rukh.eval.suite import (
    EvalConfig,
    _metric_name,
    config_path,
    evaluate,
    hub_checkpoint,
    is_hub_id,
    load_suite,
    resolve_model,
    run_suite,
)
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
    result = evaluate(checkpoint, quick_config(rukh_home), suite="quick", device="cpu")
    assert result.stage == "tiny"
    assert result.params == MoveDecoder(TOY).num_params(non_embedding=False)
    assert result.legality_argmax is None and result.legality_sampled is None
    assert (result.accuracy, result.puzzles, result.elo) == (None,) * 3
    assert result.device in ("cpu", "cuda")
    assert any("validation games not found" in note for note in result.notes)
    assert any("puzzles not found" in note for note in result.notes)
    assert any("no games were played" in note for note in result.notes)


def test_a_suite_with_games_measures_legality_and_accuracy(
    rukh_home: Path, checkpoint: Path, games: Path
) -> None:
    result, report = run_suite(
        checkpoint, quick_config(rukh_home, games), suite="quick", device="cpu"
    )
    assert result.legality_argmax is not None and 0.0 <= result.legality_argmax.rate <= 1.0
    assert result.legality_argmax.mode == "argmax"
    assert result.legality_sampled is not None and result.legality_sampled.mode == "sampled"
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


def test_a_hub_id_is_told_apart_from_a_path(checkpoint: Path) -> None:
    assert is_hub_id("chorcat/rukh-small")
    assert is_hub_id("chorcat/rukh_small.v2")
    assert not is_hub_id(str(checkpoint))
    assert not is_hub_id("checkpoints/small/best.pt")
    assert not is_hub_id("just-a-name")
    assert not is_hub_id("too/many/slashes")
    assert resolve_model(checkpoint) == checkpoint


def test_a_model_that_is_neither_a_file_nor_a_hub_id_is_an_error() -> None:
    with pytest.raises(FileNotFoundError, match="neither a checkpoint nor a Hub id"):
        resolve_model("checkpoints/does-not-exist.pt")


def test_the_cli_validates_the_model_without_touching_the_network(rukh_home: Path) -> None:
    """A bad ``--model`` fails at validation; a Hub id is accepted by its shape alone."""
    result = CliRunner().invoke(app, ["eval", "--model", "not a model", "--suite", "quick"])
    assert result.exit_code == 2
    assert "neither an existing checkpoint nor a Hub id" in result.output


def test_a_hub_repository_becomes_a_checkpoint(
    rukh_home: Path, checkpoint: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``hub_checkpoint`` folds ``config.json`` plus the weights back into a checkpoint."""
    import json as json_module

    from rukh.train import load_checkpoint

    payload = load_checkpoint(checkpoint)
    published = tmp_path / "published"
    published.mkdir()
    (published / "config.json").write_text(
        json_module.dumps({"params": 1, "step": 1000, "vocab_hash": "v", **TOY.model_dump()}),
        encoding="utf-8",
    )
    torch.save(payload["model_state"], published / "pytorch_model.bin")

    def fake_download(repo_id: str, filename: str) -> str:
        path = published / filename
        if not path.is_file():
            raise FileNotFoundError(filename)
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download, raising=False)
    out = hub_checkpoint("chorcat/rukh-tiny", out_dir=tmp_path / "hub")
    assert out.name == "chorcat__rukh-tiny.pt"
    result = evaluate(out, quick_config(rukh_home), suite="quick", device="cpu")
    assert result.params == MoveDecoder(TOY).num_params(non_embedding=False)


def test_metric_names_survive_mlflow_validation() -> None:
    """MLflow rejects `+` and `<`, and it raises after every game has been played.

    The `2600+` and `<1800` bands used to take a whole evaluation down with them at the logging
    step, hours of Stockfish games included.
    """
    assert _metric_name("top1", "2600+") == "top1/2600_"
    assert _metric_name("top3", "<1800") == "top3/_1800"
    assert _metric_name("puzzles", "1000-1500") == "puzzles/1000-1500"
    allowed = set(string.ascii_letters + string.digits + "_-. /")
    for band in ("<1800", "1800-2000", "2600+"):
        assert set(_metric_name("top1", band)) <= allowed
