"""Tests for rukh.publish.model: the staged folder, the generated card and the dry run."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from typer.testing import CliRunner

from rukh.cli import app
from rukh.models import DecoderConfig, MoveDecoder
from rukh.publish import (
    CONFIG_NAME,
    README_NAME,
    SAFETENSORS_NAME,
    TORCH_NAME,
    VOCAB_PATH,
    ModelPublishConfig,
    RunSummary,
    card_context,
    copy_onnx,
    publish_model,
    read_eval,
    render_card,
)
from rukh.tokenize.uci_vocab import UciTokenizer
from rukh.train import save_checkpoint

pytestmark = pytest.mark.unit

model_module = importlib.import_module("rukh.publish.model")

TOY = DecoderConfig(vocab_size=2030, n_layer=2, n_head=2, d_model=32, block=64)
REPO = "chorcat/rukh-tiny"
RUN = RunSummary(
    run_id="run-42",
    name="tiny-2026-09-19",
    params={"lr": "0.0006", "max_steps": "20000", "precision": "bf16"},
    metrics={"val/loss": 1.23},
)
EVAL = {
    "stage": "tiny",
    "suite": "quick",
    "date": "2026-09-19",
    "legality": {"positions": 1000, "legal": 987, "rate": 0.987},
    "accuracy": {"positions": 1000, "top1": 0.314, "top3": 0.521, "bands": []},
    "puzzles": {
        "attempted": 400,
        "solved": 40,
        "rate": 0.1,
        "bands": [{"band": "1000-1500", "attempted": 200, "solved": 30, "rate": 0.15}],
    },
    "elo": {
        "elo": 1234.0,
        "ci_low": 1174.0,
        "ci_high": 1294.0,
        "games": 160,
        "score": 0.42,
        "rungs": [],
    },
}


class FakeApi:
    """An ``HfApi`` that records what it was asked to do and talks to nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))

    def upload_folder(self, **kwargs: Any) -> None:
        self.calls.append(("upload_folder", kwargs))


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = MoveDecoder(TOY)
    return save_checkpoint(
        tmp_path / "tiny" / "best.pt",
        step=1000,
        model=model,
        optimizer=None,
        cfg={"preset": "tiny", "lr": 6e-4},
        model_cfg=TOY.model_dump(),
        vocab_hash="vocab-sha",
        git_sha="git-sha",
    )


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    fake = FakeApi()
    monkeypatch.setattr(model_module, "_api", lambda: fake)
    return fake


@pytest.fixture
def no_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local MLflow store is never touched by the tests."""
    monkeypatch.setattr(model_module, "read_run", lambda *_args, **_kwargs: RUN)


@pytest.fixture
def onnx_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "onnx"
    directory.mkdir()
    (directory / "model-fp16.onnx").write_bytes(b"fp16")
    (directory / "model-int8.onnx").write_bytes(b"int8")
    return directory


def test_a_dry_run_never_calls_the_api(
    rukh_home: Path, checkpoint: Path, api: FakeApi, no_mlflow: None
) -> None:
    result = publish_model(checkpoint, REPO, ModelPublishConfig(), dry_run=True)
    assert api.calls == []
    assert result.dry_run is True
    assert Path(result.folder) == rukh_home / "artifacts" / "publish" / REPO
    assert Path(result.card_path).is_file()


def test_a_real_run_creates_the_repo_and_uploads_the_folder(
    rukh_home: Path, checkpoint: Path, api: FakeApi, no_mlflow: None
) -> None:
    result = publish_model(checkpoint, REPO, ModelPublishConfig())
    assert [name for name, _ in api.calls] == ["create_repo", "upload_folder"]
    assert api.calls[0][1]["repo_id"] == REPO
    assert api.calls[1][1]["folder_path"] == str(Path(result.folder))
    assert api.calls[1][1]["repo_type"] == "model"


def test_the_staged_folder_holds_weights_config_and_the_vocabulary(
    rukh_home: Path, checkpoint: Path, api: FakeApi, no_mlflow: None
) -> None:
    result = publish_model(checkpoint, REPO, ModelPublishConfig(), dry_run=True)
    folder = Path(result.folder)
    assert (folder / CONFIG_NAME).is_file()
    assert (folder / VOCAB_PATH).is_file()
    assert (folder / README_NAME).is_file()
    weights = SAFETENSORS_NAME if result.weights_format == "safetensors" else TORCH_NAME
    assert (folder / weights).is_file()
    assert result.weights_format in ("safetensors", "torch")
    assert UciTokenizer.from_file(folder / VOCAB_PATH) is not None


def test_the_config_records_the_shape_and_the_provenance(
    rukh_home: Path, checkpoint: Path, api: FakeApi, no_mlflow: None
) -> None:
    result = publish_model(checkpoint, REPO, ModelPublishConfig(), dry_run=True)
    config = json.loads((Path(result.folder) / CONFIG_NAME).read_text(encoding="utf-8"))
    assert config["architectures"] == ["MoveDecoder"]
    assert config["n_layer"] == TOY.n_layer
    assert config["d_model"] == TOY.d_model
    assert config["block"] == TOY.block
    assert config["vocab_hash"] == "vocab-sha"
    assert config["git_sha"] == "git-sha"
    assert config["step"] == 1000
    assert config["params"] == result.params


def test_the_onnx_files_are_copied(
    rukh_home: Path, checkpoint: Path, api: FakeApi, no_mlflow: None, onnx_dir: Path
) -> None:
    result = publish_model(checkpoint, REPO, ModelPublishConfig(), onnx_dir=onnx_dir, dry_run=True)
    assert "onnx/model-fp16.onnx" in result.files
    assert (Path(result.folder) / "onnx" / "model-int8.onnx").read_bytes() == b"int8"


def test_an_empty_onnx_directory_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="holds none of"):
        copy_onnx(empty, tmp_path)
    with pytest.raises(FileNotFoundError, match="no ONNX directory"):
        copy_onnx(tmp_path / "absent", tmp_path)


def test_read_eval_finds_the_harness_results(rukh_home: Path) -> None:
    cfg = ModelPublishConfig()
    assert read_eval("tiny", cfg) is None
    path = rukh_home / "artifacts" / "eval" / "tiny" / "results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(EVAL), encoding="utf-8")
    assert read_eval("tiny", cfg) == EVAL


def test_the_card_carries_the_metrics_and_the_recipe() -> None:
    config = {"params": 5_000_000, "block": 200, "n_layer": 6}
    text = render_card(
        card_context("chorcat/rukh-tiny", "tiny", ModelPublishConfig(), config, EVAL, RUN, ["a"])
    )
    assert "98.7 %" in text  # legality
    assert "31.4 %" in text  # top-1
    assert "52.1 %" in text  # top-3
    assert "1234 (95 % CI 1174-1294)" in text
    assert "| 1000-1500 | 15.0 % |" in text
    assert "`max_steps` | `20000`" in text
    assert "run-42" in text
    assert "?stage=tiny" in text


def test_the_card_has_the_keys_the_hub_needs() -> None:
    config = {"params": 5_000_000, "block": 200}
    text = render_card(
        card_context("chorcat/rukh-tiny", "tiny", ModelPublishConfig(), config, EVAL, RUN, [])
    )
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---")[1])
    assert front["license"] == "apache-2.0"
    assert front["pipeline_tag"] == "text-generation"
    assert front["library_name"] == "rukh"
    assert "chess" in front["tags"]
    assert "chorcat/rukh-games-1800" in front["datasets"]


def test_a_card_without_an_evaluation_says_so() -> None:
    text = render_card(
        card_context(
            "chorcat/rukh-tiny", "tiny", ModelPublishConfig(), {"params": 1}, None, None, []
        )
    )
    assert "Not evaluated yet" in text
    assert "| Estimated Elo | n/a |" in text
    assert "was not available when the card was generated" in text


def test_the_cli_dry_run_stages_without_the_network(
    rukh_home: Path, checkpoint: Path, api: FakeApi, no_mlflow: None
) -> None:
    result = CliRunner().invoke(
        app, ["publish", "model", "--ckpt", str(checkpoint), "--repo", REPO, "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert api.calls == []
    assert "dry-run" in result.output
    assert REPO in result.output


def test_the_cli_reports_a_missing_onnx_directory(
    rukh_home: Path, checkpoint: Path, api: FakeApi, no_mlflow: None, tmp_path: Path
) -> None:
    empty = tmp_path / "empty-onnx"
    empty.mkdir()
    result = CliRunner().invoke(
        app,
        ["publish", "model", "--ckpt", str(checkpoint), "--repo", REPO, "--onnx", str(empty)],
    )
    assert result.exit_code == 1
    assert "holds none of" in result.output
