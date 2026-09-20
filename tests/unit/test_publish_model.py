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
    PARITY_NAME,
    README_NAME,
    SAFETENSORS_NAME,
    TORCH_NAME,
    VOCAB_PATH,
    ModelPublishConfig,
    RunSummary,
    acceptance_bars,
    card_context,
    copy_onnx,
    counterpart_stage,
    parity_context,
    publish_model,
    publish_state,
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
    "model_sha": "same-weights",
    "config": {"temperature": 0.6, "top_k": 20},
    "legality_argmax": {
        "positions": 1000,
        "legal": 987,
        "rate": 0.987,
        "mode": "argmax",
        "temperature": None,
        "top_k": None,
    },
    "legality_sampled": {
        "positions": 1000,
        "legal": 901,
        "rate": 0.901,
        "mode": "sampled",
        "temperature": 0.6,
        "top_k": 20,
    },
    "notes": ["the Elo interval covers sampling noise only"],
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
        "cut": 4,
        "adjudicated": 4,
        "separated": False,
        "elo_lower": None,
        "elo_upper": None,
    },
}


PARITY = {
    "version": 1,
    "kind": "decoder",
    "measures": "the argmax move",
    "positions": 1000,
    "source": "validation",
    "exporter": "dynamo",
    "warning": None,
    "precisions": {
        "fp32": {"file": "model.onnx", "agreement": 1.0, "max_abs_logit_delta": 1.2e-07},
        "fp16": {"file": "model-fp16.onnx", "agreement": 0.998, "max_abs_logit_delta": 0.02},
        "int8": {"file": "model-int8.onnx", "agreement": 0.954, "max_abs_logit_delta": 1.4},
    },
}
GREEDY = {
    **EVAL,
    "stage": "tiny-greedy",
    "model_sha": "same-weights",
    "config": {"temperature": 0.05, "top_k": 1},
    "elo": {**EVAL["elo"], "elo": 1007.0},
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
    (directory / PARITY_NAME).write_text(json.dumps(PARITY), encoding="utf-8")
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


def test_the_published_weights_hold_no_tensor_twice(
    rukh_home: Path, checkpoint: Path, api: FakeApi, no_mlflow: None
) -> None:
    """Tied weights would appear under two names, which ``safetensors.save_file`` refuses."""
    model = MoveDecoder(TOY)
    assert model.cfg.tie_embeddings
    assert model.lm_head.weight.data_ptr() == model.tokens.weight.data_ptr()

    state = publish_state(model)
    assert "lm_head.weight" not in state
    assert "tokens.weight" in state
    pointers = [tensor.data_ptr() for tensor in state.values()]
    assert len(pointers) == len(set(pointers))

    reloaded = MoveDecoder(TOY)
    reloaded.load_state_dict(state, strict=False)
    assert torch.equal(reloaded.lm_head.weight, reloaded.tokens.weight)  # re-tied on load

    result = publish_model(checkpoint, REPO, ModelPublishConfig(), dry_run=True)
    assert result.tied_embeddings is True


def test_the_parameter_count_is_the_models_own_and_counts_the_embedding_once(
    rukh_home: Path, checkpoint: Path, api: FakeApi, no_mlflow: None
) -> None:
    result = publish_model(checkpoint, REPO, ModelPublishConfig(), dry_run=True)
    model = MoveDecoder(TOY)
    assert result.params == model.num_params(non_embedding=False)

    naive = sum(int(tensor.numel()) for tensor in model.state_dict().values())
    assert naive == result.params + model.tokens.weight.numel()  # the old double count

    config = json.loads((Path(result.folder) / CONFIG_NAME).read_text(encoding="utf-8"))
    assert config["params"] == result.params
    card = Path(result.card_path).read_text(encoding="utf-8")
    assert f"{result.params:,} parameters" in card


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
    assert "98.7 %" in text  # legality, argmax
    assert "90.1 %" in text  # legality, sampled
    assert "sampled (T=0.6, top-k 20)" in text
    assert "the Elo interval covers sampling noise only" in text
    assert "31.4 %" in text  # top-1
    assert "52.1 %" in text  # top-3
    assert "1234 (95 % CI 1174-1294)" in text
    assert "| 1000-1500 | 15.0 % |" in text
    assert "`max_steps` | `20000`" in text
    assert "run-42" in text
    assert "?stage=tiny" in text


def test_the_card_explains_the_tied_head_when_the_weights_leave_it_out() -> None:
    config = {"params": 5_000_000, "block": 200, "tie_embeddings": True}
    tied = render_card(
        card_context("chorcat/rukh-tiny", "tiny", ModelPublishConfig(), config, EVAL, RUN, ["a"])
    )
    assert "`lm_head.weight`" in tied
    assert "Re-tie it after" in tied
    untied = render_card(
        card_context(
            "chorcat/rukh-tiny",
            "tiny",
            ModelPublishConfig(),
            {"params": 1, "tie_embeddings": False},
            EVAL,
            RUN,
            ["a"],
        )
    )
    assert "`lm_head.weight`" not in untied


def test_a_separated_elo_is_shown_as_a_one_sided_bound() -> None:
    separated = {
        **EVAL,
        "elo": {
            **EVAL["elo"],
            "ci_low": None,
            "ci_high": None,
            "separated": True,
            "elo_lower": 2450.0,
        },
    }
    text = render_card(
        card_context(
            "chorcat/rukh-tiny", "tiny", ModelPublishConfig(), {"params": 1}, separated, RUN, []
        )
    )
    assert "> 2450 (one-sided 95 % bound; every game won)" in text


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


def test_the_card_puts_every_acceptance_bar_next_to_what_was_measured() -> None:
    """The bars are the project's own, from ``GOAL.md``, and the verdict is computed here."""
    text = render_card(
        card_context(
            "chorcat/rukh-tiny", "tiny", ModelPublishConfig(), {"params": 1}, EVAL, RUN, []
        )
    )
    assert "### Acceptance bars" in text
    assert "| Legality without the mask, argmax | at least 99 % | 98.7 % | **not met** |" in text
    assert "| Estimated Elo | at least 1200 | 1234 (95 % CI 1174-1294) | met |" in text
    assert "**The legality bar is not met**" in text
    assert "**The Elo bar is not met**" not in text


def test_a_bar_that_is_not_met_is_said_in_words_with_its_reason() -> None:
    """A table cell is not an explanation: the card has to say why, not only that."""
    short = {**EVAL, "elo": {**EVAL["elo"], "elo": 1007.0}}
    text = render_card(
        card_context(
            "chorcat/rukh-tiny", "tiny", ModelPublishConfig(), {"params": 1}, short, RUN, []
        )
    )
    assert "The Elo bar is not met" in text
    assert "The legality bar is not met" in text
    # Not only that it failed: why, and where the reader should look instead. The wording moves
    # with the project -- it used to blame a 5.9 M-game corpus -- so pin the substance.
    assert "rukh-medium-dpo" in text and "baseline" in text
    assert "19.0 M games" in text
    # And every card says the ratings replaced lower published ones, met or not.
    assert "replace lower ones" in text and "428 and 581" in text


def test_the_bars_are_omitted_rather_than_guessed_when_nothing_was_measured() -> None:
    assert acceptance_bars(None) == []
    assert acceptance_bars({"elo": {"elo": 1300.0}}) == [
        {
            "id": "elo",
            "name": "Estimated Elo",
            "target": "at least 1200",
            "measured": "1300 (no interval)",
            "met": True,
        }
    ]
    text = render_card(
        card_context(
            "chorcat/rukh-tiny", "tiny", ModelPublishConfig(), {"params": 1}, None, None, []
        )
    )
    assert "### Acceptance bars" not in text


def test_the_card_gives_the_measured_quantization_parity_and_what_it_means() -> None:
    text = render_card(
        card_context(
            "chorcat/rukh-tiny",
            "tiny",
            ModelPublishConfig(),
            {"params": 1},
            EVAL,
            RUN,
            ["onnx/model.onnx"],
            PARITY,
        )
    )
    assert "1000\nvalidation positions" in text
    assert "| `model.onnx` (fp32) | 100.0 % | 1.2e-07 |" in text
    assert "| `model-int8.onnx` (int8) | 95.4 % | 1.4 |" in text
    assert "it picks a different move in 4.6 % of\npositions, roughly one in 22" in text
    assert "the WASM fallback loads" in text
    assert "`onnx/parity.json`" in text


def test_a_card_without_a_parity_file_says_nothing_about_parity() -> None:
    assert parity_context(None) is None
    assert parity_context({"precisions": {}}) is None
    text = render_card(
        card_context(
            "chorcat/rukh-tiny",
            "tiny",
            ModelPublishConfig(),
            {"params": 1},
            EVAL,
            RUN,
            ["onnx/model.onnx"],
        )
    )
    assert "How faithful the ONNX files are" not in text


def test_the_card_states_that_an_elo_belongs_to_the_pair_model_and_sampling() -> None:
    """D-047: one checkpoint, two sampling settings, two ratings more than 200 points apart."""
    text = render_card(
        card_context(
            "chorcat/rukh-tiny",
            "tiny",
            ModelPublishConfig(),
            {"params": 1},
            EVAL,
            RUN,
            [],
            None,
            GREEDY,
        )
    )
    assert "### An Elo belongs to the pair model+sampling" in text
    assert "**1007 Elo** at\ntemperature 0.05, top-k 1" in text
    assert "**1234 Elo** at temperature 0.6, top-k 20" in text
    assert "temperature 0.6, top-k 20 point" in text  # the table's own operating point


def test_two_evaluations_of_different_weights_are_never_called_the_same_model() -> None:
    other = {**GREEDY, "model_sha": "other-weights"}
    text = render_card(
        card_context(
            "chorcat/rukh-tiny",
            "tiny",
            ModelPublishConfig(),
            {"params": 1},
            EVAL,
            RUN,
            [],
            None,
            other,
        )
    )
    assert "An Elo belongs to the pair" not in text


def test_the_counterpart_stage_is_the_other_sampling_point() -> None:
    assert counterpart_stage("small") == "small-greedy"
    assert counterpart_stage("small-greedy") == "small"


def test_the_parity_file_is_published_with_the_onnx_files(
    rukh_home: Path, checkpoint: Path, api: FakeApi, no_mlflow: None, onnx_dir: Path
) -> None:
    """It is small, it is the evidence behind the card's numbers, so it goes to the Hub too."""
    result = publish_model(checkpoint, REPO, ModelPublishConfig(), onnx_dir=onnx_dir, dry_run=True)
    assert f"onnx/{PARITY_NAME}" in result.files
    assert (Path(result.folder) / "onnx" / PARITY_NAME).is_file()
    card = Path(result.card_path).read_text(encoding="utf-8")
    assert "| `model-int8.onnx` (int8) | 95.4 % |" in card


def test_publish_refuses_another_models_numbers(tmp_path: Path) -> None:
    """A card must never carry metrics measured on different weights.

    `read_eval` finds results by stage name alone, so the wrong `--stage` silently renders a card
    whose Elo, legality and puzzle rates belong to another checkpoint. It nearly shipped twice:
    once for the encoder (D-061) and once publishing `small-v3` under `small` v1's stage.
    """
    from rukh.eval.cache import file_sha
    from rukh.publish.model import check_eval_matches

    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights that were published")
    other = tmp_path / "other.pt"
    other.write_bytes(b"weights that were measured")

    # Same checkpoint: allowed, and the sha is the file's own.
    check_eval_matches(ckpt, "small-greedy", {"model_sha": file_sha(ckpt)})
    # No evaluation at all, or one without a sha: nothing to contradict.
    check_eval_matches(ckpt, "small-greedy", None)
    check_eval_matches(ckpt, "small-greedy", {})

    with pytest.raises(ValueError, match="measured on a different checkpoint"):
        check_eval_matches(ckpt, "small-greedy", {"model_sha": file_sha(other)})
