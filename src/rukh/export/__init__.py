"""Export: ONNX for the browser, fp16 and int8 quantization and PyTorch parity.

``export_all`` is the whole of ``rukh export``: write the fp32 graph, derive the fp16 and int8
files the two browser backends need, and check every file it produced against PyTorch. It
exports two kinds of model - the decoder's next move and the encoder's ``value`` and ``blunder``
heads - through the same path, with one wrapper and one parity function each.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rukh.data.labels import LabelsConfig
from rukh.export.embed import EmbedResult, embed_positions, fen4
from rukh.export.onnx import (
    BLUNDER_OUTPUT,
    DEFAULT_OPSET,
    ENCODER_OUTPUTS,
    INPUT_NAME,
    MODEL_NAME,
    OUTPUT_NAME,
    VALUE_OUTPUT,
    EncoderHeads,
    ExportResult,
    LastStepLogits,
    clear_value_info,
    export_encoder_onnx,
    export_onnx,
    read_metadata,
    sequence_lengths,
    set_metadata,
    target_path,
    verify_dynamic_seq,
    write_metadata,
)
from rukh.export.parity import (
    DEFAULT_GAMES,
    EncoderParityResult,
    ParityResult,
    encoder_parity,
    encoder_parity_positions,
    label_positions,
    parity,
    parity_positions,
    random_prefixes,
    validation_prefixes,
)
from rukh.export.quantize import (
    FP16_NAME,
    INT8_NAME,
    QUANTIZED_OPS,
    QuantizeResult,
    quantize_int8,
    to_fp16,
)

KINDS = ("decoder", "encoder")
"""What ``rukh export --kind`` accepts."""


class ExportBundle(BaseModel):
    """Everything one ``rukh export`` produced."""

    model_config = ConfigDict(extra="forbid")

    onnx: ExportResult
    fp16: QuantizeResult | None = None
    int8: QuantizeResult | None = None
    parity: dict[str, ParityResult] = {}
    """Parity per file kind: ``fp32``, ``fp16``, ``int8``."""
    heads_parity: dict[str, EncoderParityResult] = {}
    """The encoder's parity per file kind: the blunder decision and the drift of ``value``."""
    parity_source: str | None = None
    """``validation`` (the positions of ``docs/spec/02`` §6), ``random-walk``, or, for the
    encoder, ``validation-labels``."""
    parity_warning: str | None = None


def _quantize(bundle: ExportBundle, fp16: bool, int8: bool) -> dict[str, str]:
    """Derive the browser's files from the fp32 graph and return every file to check."""
    if fp16:
        bundle.fp16 = to_fp16(Path(bundle.onnx.path))
    if int8:
        bundle.int8 = quantize_int8(Path(bundle.onnx.path))
    # The demo loads the fp16 or the int8 file, not the fp32 one, so the context length has to
    # travel with them too; neither converter promises to keep the metadata of its input.
    for quantized in (bundle.fp16, bundle.int8):
        if quantized is not None and bundle.onnx.metadata:
            set_metadata(Path(quantized.path), bundle.onnx.metadata)
    checks = {"fp32": bundle.onnx.path}
    if bundle.fp16 is not None:
        checks["fp16"] = bundle.fp16.path
    if bundle.int8 is not None:
        checks["int8"] = bundle.int8.path
    return checks


def export_all(
    ckpt: Path,
    out: Path,
    kind: str = "decoder",
    opset: int = DEFAULT_OPSET,
    seq_len: int = 200,
    fp16: bool = False,
    int8: bool = False,
    check_parity: bool = False,
    positions: int = 1_000,
    seed: int = 0,
    games: Path | None = None,
    labels: LabelsConfig | None = None,
    split: str = "val",
) -> ExportBundle:
    """Export, quantize and check one checkpoint in a single pass.

    For the decoder, parity is measured on validation positions (``games``, or the validation
    month of P1 when it is on disk) and falls back to random legal walks with a warning when
    there are none. For the encoder it is measured on the held-out labelled positions of
    ``rukh.data.labels`` and is skipped, with a warning, when those have not been built: a
    random board is not a position the demo will ever be asked to evaluate.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {', '.join(KINDS)}")
    if kind == "encoder":
        return _export_encoder(ckpt, out, opset, fp16, int8, check_parity, positions, labels, split)
    from rukh.tokenize.uci_vocab import UciTokenizer
    from rukh.train import load_model

    model, _payload = load_model(Path(ckpt))
    bundle = ExportBundle(onnx=export_onnx(model, out, opset=opset, seq_len=seq_len))
    checks = _quantize(bundle, fp16, int8)
    if check_parity:
        prefixes, source, warning = parity_positions(
            UciTokenizer(), n=positions, block=model.cfg.block, seed=seed, games=games
        )
        bundle.parity_source = source
        bundle.parity_warning = warning
        bundle.parity = {
            name: parity(model, Path(path), prefixes, n=positions) for name, path in checks.items()
        }
    return bundle


def _export_encoder(
    ckpt: Path,
    out: Path,
    opset: int,
    fp16: bool,
    int8: bool,
    check_parity: bool,
    positions: int,
    labels: LabelsConfig | None,
    split: str,
) -> ExportBundle:
    """``export_all(kind="encoder")``: the two heads, the same files, its own parity."""
    from rukh.train import load_heads

    model, _payload = load_heads(Path(ckpt))
    bundle = ExportBundle(onnx=export_encoder_onnx(model, out, opset=opset))
    checks = _quantize(bundle, fp16, int8)
    if check_parity:
        items, source, warning = encoder_parity_positions(labels, n=positions, split=split)
        bundle.parity_source = source
        bundle.parity_warning = warning
        if items:
            bundle.heads_parity = {
                name: encoder_parity(model, Path(path), items, n=positions)
                for name, path in checks.items()
            }
    return bundle


__all__ = [
    "BLUNDER_OUTPUT",
    "DEFAULT_GAMES",
    "DEFAULT_OPSET",
    "ENCODER_OUTPUTS",
    "FP16_NAME",
    "INPUT_NAME",
    "INT8_NAME",
    "KINDS",
    "MODEL_NAME",
    "OUTPUT_NAME",
    "QUANTIZED_OPS",
    "VALUE_OUTPUT",
    "EmbedResult",
    "EncoderHeads",
    "EncoderParityResult",
    "ExportBundle",
    "ExportResult",
    "LastStepLogits",
    "ParityResult",
    "QuantizeResult",
    "clear_value_info",
    "embed_positions",
    "encoder_parity",
    "encoder_parity_positions",
    "export_all",
    "export_encoder_onnx",
    "export_onnx",
    "fen4",
    "label_positions",
    "parity",
    "parity_positions",
    "quantize_int8",
    "random_prefixes",
    "read_metadata",
    "sequence_lengths",
    "set_metadata",
    "target_path",
    "to_fp16",
    "validation_prefixes",
    "verify_dynamic_seq",
    "write_metadata",
]
