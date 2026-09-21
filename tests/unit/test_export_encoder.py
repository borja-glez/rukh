"""Tests for the encoder's export, its embeddings and its publication.

The decoder's own export tests (``test_export.py``) are untouched on purpose: the two graphs go
out through the same code, and the decoder's path must keep behaving exactly as it did.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch
from typer.testing import CliRunner

from rukh.cli import app
from rukh.export import (
    BLUNDER_OUTPUT,
    ENCODER_OUTPUTS,
    INPUT_NAME,
    PARITY_NAME,
    VALUE_OUTPUT,
    EncoderHeads,
    embed_positions,
    encoder_parity,
    encoder_parity_positions,
    export_all,
    export_encoder_onnx,
    fen4,
    quantize_int8,
    read_metadata,
    to_fp16,
)
from rukh.models import EncoderConfig, PositionEncoder
from rukh.models.heads import MultiHead
from rukh.models.squares import SQUARE_TOKENS, fen_to_tokens
from rukh.publish import CONFIG_NAME, VOCAB_PATH, ModelPublishConfig, publish_model
from rukh.train import save_checkpoint

pytestmark = pytest.mark.unit

TOY = EncoderConfig(input="squares", n_layer=2, n_head=2, d_model=32, dropout=0.0)
HAS_ONNX = all(importlib.util.find_spec(name) for name in ("onnx", "onnxruntime"))
requires_onnx = pytest.mark.skipif(not HAS_ONNX, reason="onnx and onnxruntime are not installed")
FENS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
    "r1bqkbnr/pppp1ppp/2n5/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq -",
    "r1bqkbnr/pppp1Qpp/2n5/4p3/4P3/8/PPPP1PPP/RNB1KBNR b KQkq -",
)


@pytest.fixture(scope="module")
def model() -> MultiHead:
    torch.manual_seed(0)
    return MultiHead(PositionEncoder(TOY)).eval()


@pytest.fixture(scope="module")
def positions() -> list[list[int]]:
    return [fen_to_tokens(fen) for fen in FENS]


@pytest.fixture(scope="module")
def exported(model: MultiHead, tmp_path_factory: pytest.TempPathFactory) -> Any:
    return export_encoder_onnx(model, tmp_path_factory.mktemp("encoder-onnx"))


def checkpoint(path: Path, model: MultiHead, **cfg: Any) -> Path:
    save_checkpoint(
        path,
        step=7,
        model=model,
        optimizer=None,
        cfg={
            "pooling": "mean",
            "weights": {"value": 1.0, "blunder": 1.0, "result": 0.5},
            **cfg,
        },
        model_cfg=TOY.model_dump(),
    )
    return path


# --- the wrapper and the graph ----------------------------------------------------------------


def test_the_wrapper_returns_the_two_heads_the_demo_needs(
    model: MultiHead, positions: list[list[int]]
) -> None:
    idx = torch.tensor(positions, dtype=torch.long)
    with torch.no_grad():
        value, blunder = EncoderHeads(model)(idx)
        reference = model(idx)
    assert value.shape == blunder.shape == (len(positions),)
    assert torch.allclose(value, reference["value"])
    assert torch.allclose(blunder, torch.sigmoid(reference["blunder"]))
    assert bool(((blunder >= 0.0) & (blunder <= 1.0)).all())  # a probability, not a logit


@requires_onnx
def test_the_exported_encoder_matches_pytorch_on_both_outputs(
    model: MultiHead, positions: list[list[int]], exported: Any
) -> None:
    assert exported.exporter == "dynamo", exported.warning
    assert exported.outputs == list(ENCODER_OUTPUTS)
    result = encoder_parity(model, Path(exported.path), positions)
    assert result.agreement == 1.0
    assert result.max_abs_value_delta < 1e-4
    assert result.max_abs_blunder_delta < 1e-4
    assert result.mismatches == []


@requires_onnx
def test_the_exported_encoder_runs_under_onnxruntime_in_batches(
    model: MultiHead, positions: list[list[int]], exported: Any
) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(exported.path, providers=["CPUExecutionProvider"])
    assert [output.name for output in session.get_outputs()] == list(ENCODER_OUTPUTS)
    for batch in (1, len(positions)):
        idx = np.asarray(positions[:batch], dtype=np.int64)
        value, blunder = session.run(None, {INPUT_NAME: idx})
        assert value.shape == blunder.shape == (batch,)
        with torch.no_grad():
            expected = EncoderHeads(model)(torch.from_numpy(idx))
        assert np.allclose(value, expected[0].numpy(), atol=1e-4)
        assert np.allclose(blunder, expected[1].numpy(), atol=1e-4)


@requires_onnx
def test_the_metadata_says_what_the_file_is(exported: Any) -> None:
    metadata = read_metadata(Path(exported.path))
    assert metadata["rukh_kind"] == "encoder"
    assert metadata["rukh_heads"] == f"{VALUE_OUTPUT},{BLUNDER_OUTPUT}"
    assert metadata["rukh_input"] == "squares"
    assert metadata["rukh_blunder"] == "probability"
    assert metadata["rukh_block"] == str(SQUARE_TOKENS)
    assert exported.metadata == metadata


def test_the_squares_scheme_has_no_sequence_axis_to_make_dynamic(exported: Any) -> None:
    # 69 tokens, always: there is nothing to grow, so the axis is fixed and says so.
    assert exported.seq_len == SQUARE_TOKENS
    assert exported.dynamic_seq is False
    assert exported.dynamic_batch is True


@requires_onnx
def test_the_int8_file_is_smaller_than_the_fp32_one(exported: Any) -> None:
    quantized = quantize_int8(Path(exported.path))
    assert Path(quantized.path).is_file()
    assert quantized.bytes < exported.bytes


@requires_onnx
def test_the_fp16_file_is_smaller_and_still_agrees(
    model: MultiHead, positions: list[list[int]], exported: Any
) -> None:
    half = to_fp16(Path(exported.path))
    assert half.bytes < exported.bytes
    result = encoder_parity(model, Path(half.path), positions)
    assert result.agreement == 1.0
    assert result.max_abs_value_delta < 1e-2


@requires_onnx
def test_export_all_writes_the_three_files_and_keeps_the_metadata(
    model: MultiHead, tmp_path: Path
) -> None:
    bundle = export_all(
        checkpoint(tmp_path / "run" / "best.pt", model),
        tmp_path / "onnx",
        kind="encoder",
        fp16=True,
        int8=True,
    )
    assert bundle.onnx.kind == "encoder"
    assert bundle.fp16 is not None and bundle.int8 is not None
    for quantized in (bundle.fp16, bundle.int8):
        assert read_metadata(Path(quantized.path))["rukh_kind"] == "encoder"


@requires_onnx
def test_export_all_measures_parity_on_the_held_out_labels(
    model: MultiHead, tmp_path: Path
) -> None:
    from helpers_labels import source_frame
    from rukh.data.labels import LabelsConfig

    source = tmp_path / "positions-eval.parquet"
    source_frame().write_parquet(source)
    bundle = export_all(
        checkpoint(tmp_path / "run" / "best.pt", model),
        tmp_path / "onnx",
        kind="encoder",
        int8=True,
        check_parity=True,
        positions=8,
        labels=LabelsConfig(positions_eval=str(source), val_fraction=0.999),
    )
    assert bundle.parity_source == "validation-labels" and bundle.parity_warning is None
    assert set(bundle.heads_parity) == {"fp32", "int8"}
    assert bundle.parity == {}  # the decoder's kind of parity is not measured here
    assert bundle.heads_parity["fp32"].agreement == 1.0
    assert bundle.heads_parity["fp32"].max_abs_value_delta < 1e-4
    assert bundle.heads_parity["fp32"].positions == 8

    written = tmp_path / "onnx" / PARITY_NAME
    assert bundle.parity_path == written.as_posix()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["kind"] == "encoder"
    assert payload["measures"] == "the blunder decision at p >= 0.5"
    assert payload["source"] == "validation-labels" and payload["positions"] == 8
    assert set(payload["precisions"]) == {"fp32", "int8"}
    assert payload["precisions"]["int8"]["file"] == "model-int8.onnx"
    assert payload["precisions"]["fp32"]["max_abs_blunder_delta"] is not None


def test_an_unknown_kind_is_an_error(tmp_path: Path, model: MultiHead) -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        export_all(checkpoint(tmp_path / "run" / "best.pt", model), tmp_path, kind="engine")


def test_the_command_line_refuses_an_unknown_kind(tmp_path: Path, model: MultiHead) -> None:
    invocation = CliRunner().invoke(
        app,
        [
            "export",
            "--ckpt",
            str(checkpoint(tmp_path / "run" / "best.pt", model)),
            "--out",
            str(tmp_path / "onnx"),
            "--kind",
            "engine",
        ],
    )
    assert invocation.exit_code == 2
    assert "--kind must be one of decoder, encoder" in invocation.output


@requires_onnx
def test_the_command_line_exports_the_encoder(tmp_path: Path, model: MultiHead) -> None:
    invocation = CliRunner().invoke(
        app,
        [
            "export",
            "--ckpt",
            str(checkpoint(tmp_path / "run" / "best.pt", model)),
            "--out",
            str(tmp_path / "onnx"),
            "--kind",
            "encoder",
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    assert "kind:     encoder -> value, blunder" in invocation.output
    assert (tmp_path / "onnx" / "model.onnx").is_file()


# --- parity on validation positions, never on random boards ------------------------------------


def test_the_encoder_parity_refuses_to_invent_positions(
    tmp_path: Path, rukh_home: Path, model: MultiHead
) -> None:
    from rukh.data.labels import LabelsConfig

    items, source, warning = encoder_parity_positions(
        model, LabelsConfig(positions_eval=str(tmp_path / "absent.parquet"))
    )
    assert items == [] and source == "none"
    assert warning is not None and "skipped rather than measured" in warning


def test_the_encoder_parity_reads_the_held_out_labels(tmp_path: Path, model: MultiHead) -> None:
    from helpers_labels import source_frame
    from rukh.data.labels import LabelsConfig

    source = tmp_path / "positions-eval.parquet"
    source_frame().write_parquet(source)
    cfg = LabelsConfig(positions_eval=str(source), val_fraction=0.999)
    items, label, warning = encoder_parity_positions(model, cfg, n=5)
    assert label == "validation-labels" and warning is None
    assert len(items) == 5
    assert all(len(tokens) == SQUARE_TOKENS for tokens in items)


def test_parity_without_positions_is_an_error(model: MultiHead) -> None:
    with pytest.raises(ValueError, match="at least one position"):
        encoder_parity(model, Path("unused.onnx"), [])


# --- embeddings ---------------------------------------------------------------------------------


def test_the_embeddings_line_up_with_their_sidecar(tmp_path: Path, model: MultiHead) -> None:
    source = tmp_path / "positions.parquet"
    pl.DataFrame({"fen": list(FENS)}).write_parquet(source)
    result = embed_positions(
        checkpoint(tmp_path / "run" / "best.pt", model),
        source,
        tmp_path / "embeddings.npy",
        batch_size=2,
        device="cpu",
    )
    array = np.load(result.array)
    sidecar = pl.read_parquet(result.sidecar)
    assert array.shape == (len(FENS), TOY.d_model)
    assert sidecar["row"].to_list() == list(range(len(FENS)))
    assert sidecar["fen4"].to_list() == [fen4(fen) for fen in FENS]
    # Row by row, the array is the mean-pooled representation of the FEN the sidecar names.
    with torch.no_grad():
        expected = model.encoder.pool(
            torch.tensor([fen_to_tokens(fen) for fen in FENS], dtype=torch.long), "mean"
        )
    assert np.allclose(array, expected.numpy(), atol=1e-5)


def test_embedding_a_decoder_checkpoint_is_refused(tmp_path: Path) -> None:
    from rukh.models import DecoderConfig, MoveDecoder

    decoder = MoveDecoder(DecoderConfig(vocab_size=2030, n_layer=1, n_head=2, d_model=32, block=16))
    path = tmp_path / "decoder" / "best.pt"
    save_checkpoint(
        path,
        step=1,
        model=decoder,
        optimizer=None,
        cfg={},
        model_cfg=decoder.cfg.model_dump(),
    )
    source = tmp_path / "positions.parquet"
    pl.DataFrame({"fen": list(FENS)}).write_parquet(source)
    with pytest.raises(ValueError, match="embeddings need an encoder"):
        embed_positions(path, source, tmp_path / "out.npy", device="cpu")


def test_the_embed_command_writes_both_files(tmp_path: Path, model: MultiHead) -> None:
    source = tmp_path / "positions.parquet"
    pl.DataFrame({"fen": list(FENS)}).write_parquet(source)
    invocation = CliRunner().invoke(
        app,
        [
            "encoder",
            "embed",
            "--positions",
            str(source),
            "--out",
            str(tmp_path / "vectors.npy"),
            "--ckpt",
            str(checkpoint(tmp_path / "run" / "best.pt", model)),
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--json",
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    result = json.loads(invocation.output)
    assert result["positions"] == len(FENS) and result["pooling"] == "mean"
    assert Path(result["array"]).is_file() and Path(result["sidecar"]).is_file()


# --- publication ---------------------------------------------------------------------------------


def test_publishing_the_encoder_stages_a_card_with_both_f1_values(
    tmp_path: Path, rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    model_module = importlib.import_module("rukh.publish.model")

    def refuse() -> Any:  # pragma: no cover - the point is that it is never called
        raise AssertionError("a dry run must not touch the Hub")

    monkeypatch.setattr(model_module, "_api", refuse)
    monkeypatch.setattr(model_module, "read_run", lambda *args, **kwargs: None)
    torch.manual_seed(0)
    encoder = MultiHead(PositionEncoder(TOY)).eval()
    evaluation = {
        "stage": "encoder",
        "date": "2026-09-19",
        "items": 900,
        "blunder_items": 800,
        "encoder_blunder": {"name": "encoder", "f1": 0.42, "precision": 0.40, "recall": 0.44},
        "heuristic_blunder": {"name": "heuristic", "f1": 0.31},
        "encoder_value": {"name": "encoder", "pearson": 0.83, "spearman": 0.81},
        "f1_margin": 11.0,
        "result_accuracy": 0.55,
        "label_curve": [{"fraction": 0.1, "train_labels": 100}],
        "notes": ["the split is drawn by game_id"],
    }
    results = rukh_home / "artifacts" / "eval" / "encoder" / "results.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(evaluation), encoding="utf-8")

    result = publish_model(
        checkpoint(tmp_path / "encoder-heads" / "best.pt", encoder),
        "chorcat/rukh-encoder",
        ModelPublishConfig(),
        dry_run=True,
    )
    assert result.kind == "encoder" and result.dry_run is True
    card = Path(result.card_path).read_text(encoding="utf-8")
    assert "| Blunder F1 | 42.0 % |" in card
    assert "| Blunder F1, material baseline | 31.0 % |" in card
    assert "+11.0 F1 points" in card
    assert "material" in card and "mobility" in card  # the baseline is described, not just named
    assert "69 fixed tokens" in card  # the input scheme
    assert "Apache License 2.0" in card and "APACHE-2.0" in card
    config = json.loads((Path(result.folder) / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["PositionEncoder"]
    assert config["model_type"] == "rukh-position-encoder"
    assert config["heads"] == ["value", "blunder", "result"]
    assert config["input"] == "squares"
    vocab = json.loads((Path(result.folder) / VOCAB_PATH).read_text(encoding="utf-8"))
    assert vocab["scheme"] == "squares" and vocab["sequence"] == SQUARE_TOKENS
    assert vocab["tokens"][0] == "<pad>"
    # This run had no `encoder_ckpt`, so the card must not claim a pretraining stage that did
    # not happen; the claim is gated on the checkpoint the heads actually started from.
    assert config["pretrained_from"] is None
    assert "pretrained with masked move modeling" not in card
    assert "not** pretrained" in card

    pretrained = publish_model(
        checkpoint(
            tmp_path / "encoder-heads-mmm" / "best.pt",
            encoder,
            encoder_ckpt="checkpoints/encoder-mmm/best.pt",
        ),
        "chorcat/rukh-encoder-mmm",
        ModelPublishConfig(),
        dry_run=True,
    )
    card = Path(pretrained.card_path).read_text(encoding="utf-8")
    assert "pretrained with masked move modeling" in card
    assert "checkpoints/encoder-mmm/best.pt" in card


def test_the_publish_command_accepts_an_encoder(
    tmp_path: Path, rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    model_module = importlib.import_module("rukh.publish.model")
    monkeypatch.setattr(model_module, "read_run", lambda *args, **kwargs: None)
    torch.manual_seed(0)
    encoder = MultiHead(PositionEncoder(TOY)).eval()
    invocation = CliRunner().invoke(
        app,
        [
            "publish",
            "model",
            "--ckpt",
            str(checkpoint(tmp_path / "encoder-heads" / "best.pt", encoder)),
            "--repo",
            "chorcat/rukh-encoder",
            "--dry-run",
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    assert "chorcat/rukh-encoder (model, encoder)" in invocation.output
    assert "dry-run" in invocation.output


def test_a_bare_pretrained_encoder_gets_the_pretraining_card(
    tmp_path: Path, rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A masked-move checkpoint has no heads, so none of the heads' claims may appear on it.

    It is published so a reader can skip the pretraining lab and still fine-tune the heads from
    exactly what the course used; the card has to say that, and report the one thing the run
    measured (the held-out masked-move loss ``best.pt`` was selected on).
    """
    import importlib

    from rukh.publish import RunSummary

    model_module = importlib.import_module("rukh.publish.model")
    run = RunSummary(run_id="mmm-run", params={"max_steps": "16000"}, metrics={"val/top1": 0.814})
    monkeypatch.setattr(model_module, "read_run", lambda *args, **kwargs: run)
    torch.manual_seed(0)
    encoder = PositionEncoder(TOY).eval()
    path = tmp_path / "encoder-mmm-v4" / "best.pt"
    save_checkpoint(
        path,
        step=16000,
        model=encoder,
        optimizer=None,
        cfg={"tokens_dir": "data/tokens-v4/uci"},
        model_cfg=TOY.model_dump(),
        best_val=0.6981,
    )
    result = publish_model(
        path, "chorcat/rukh-encoder-mmm", ModelPublishConfig(), stage="encoder-mmm-v4", dry_run=True
    )
    assert result.kind == "encoder"
    card = Path(result.card_path).read_text(encoding="utf-8")
    assert "pretrained only" in card
    assert "| Masked-move loss on held-out games | 0.6981 |" in card
    assert "| Masked-move top-1 on held-out games | 81.4 % |" in card
    assert "rukh pull encoder-mmm-v4" in card
    assert "| `max_steps` | `16000` |" in card
    assert "Stockfish labels, with no pretraining" not in card
    assert "Blunder F1" not in card
    config = json.loads((Path(result.folder) / CONFIG_NAME).read_text(encoding="utf-8"))
    assert config["architectures"] == ["PositionEncoder"] and config["input"] == "squares"


def test_the_encoder_card_states_its_bars_and_the_parity_of_the_three_precisions(
    tmp_path: Path, rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bar met and one not, and the parity that makes the decoder's number legible."""
    import importlib

    from rukh.export import PARITY_NAME

    model_module = importlib.import_module("rukh.publish.model")
    monkeypatch.setattr(model_module, "read_run", lambda *args, **kwargs: None)
    torch.manual_seed(0)
    encoder = MultiHead(PositionEncoder(TOY)).eval()
    evaluation = {
        "stage": "encoder",
        "date": "2026-09-19",
        "items": 10_000,
        "blunder_items": 7_453,
        "encoder_blunder": {"name": "encoder", "f1": 0.179, "precision": 0.157, "recall": 0.208},
        "heuristic_blunder": {"name": "heuristic", "f1": 0.089},
        "encoder_value": {"name": "encoder", "pearson": 0.442, "spearman": 0.407},
        "f1_margin": 9.0,
        "result_accuracy": 0.499,
    }
    results = rukh_home / "artifacts" / "eval" / "encoder" / "results.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(evaluation), encoding="utf-8")
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model-int8.onnx").write_bytes(b"int8")
    (onnx_dir / PARITY_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "encoder",
                "measures": "the blunder decision at p >= 0.5",
                "positions": 1_000,
                "source": "validation-labels",
                "exporter": "dynamo",
                "precisions": {
                    "fp32": {"file": "model.onnx", "agreement": 1.0, "max_abs_value_delta": 2e-06},
                    "int8": {
                        "file": "model-int8.onnx",
                        "agreement": 1.0,
                        "max_abs_value_delta": 0.03,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = publish_model(
        checkpoint(tmp_path / "encoder-heads" / "best.pt", encoder),
        "chorcat/rukh-encoder",
        ModelPublishConfig(),
        onnx_dir=onnx_dir,
        dry_run=True,
    )
    card = Path(result.card_path).read_text(encoding="utf-8")
    assert "### Acceptance bars" in card
    assert "material baseline | at least +5 F1 points | +9.0 F1 points | met |" in card
    assert "| Value vs Stockfish cp, Spearman | at least 0.80 | 0.407 | **not met** |" in card
    assert "**The value bar is not met.**" in card
    assert "4 000 steps" in card and "9.8 %" in card
    assert "1000\nheld-out labelled positions" in card
    assert "| `model-int8.onnx` (int8) | 100.0 % | 0.03 |" in card
    assert "Every precision, int8 included, makes the same call" in card
