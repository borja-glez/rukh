"""Export: ONNX for the browser, fp16 and int8 quantization and PyTorch parity.

``export_all`` is the whole of ``rukh export``: write the fp32 graph, derive the fp16 and int8
files the two browser backends need, and check every file it produced against PyTorch.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rukh.export.onnx import (
    DEFAULT_OPSET,
    INPUT_NAME,
    MODEL_NAME,
    OUTPUT_NAME,
    ExportResult,
    LastStepLogits,
    export_onnx,
    target_path,
)
from rukh.export.parity import ParityResult, parity, random_prefixes
from rukh.export.quantize import (
    FP16_NAME,
    INT8_NAME,
    QUANTIZED_OPS,
    QuantizeResult,
    quantize_int8,
    to_fp16,
)


class ExportBundle(BaseModel):
    """Everything one ``rukh export`` produced."""

    model_config = ConfigDict(extra="forbid")

    onnx: ExportResult
    fp16: QuantizeResult | None = None
    int8: QuantizeResult | None = None
    parity: dict[str, ParityResult] = {}
    """Parity per file kind: ``fp32``, ``fp16``, ``int8``."""


def export_all(
    ckpt: Path,
    out: Path,
    opset: int = DEFAULT_OPSET,
    seq_len: int = 200,
    fp16: bool = False,
    int8: bool = False,
    check_parity: bool = False,
    parity_positions: int = 1_000,
    seed: int = 0,
) -> ExportBundle:
    """Export, quantize and check one checkpoint in a single pass."""
    from rukh.tokenize.uci_vocab import UciTokenizer
    from rukh.train import load_model

    model, _payload = load_model(Path(ckpt))
    bundle = ExportBundle(onnx=export_onnx(model, out, opset=opset, seq_len=seq_len))
    if fp16:
        bundle.fp16 = to_fp16(Path(bundle.onnx.path))
    if int8:
        bundle.int8 = quantize_int8(Path(bundle.onnx.path))
    if check_parity:
        prefixes = random_prefixes(UciTokenizer(), parity_positions, seed=seed)
        checks = {"fp32": bundle.onnx.path}
        if bundle.fp16 is not None:
            checks["fp16"] = bundle.fp16.path
        if bundle.int8 is not None:
            checks["int8"] = bundle.int8.path
        bundle.parity = {
            kind: parity(model, Path(path), prefixes, n=parity_positions)
            for kind, path in checks.items()
        }
    return bundle


__all__ = [
    "DEFAULT_OPSET",
    "FP16_NAME",
    "INPUT_NAME",
    "INT8_NAME",
    "MODEL_NAME",
    "OUTPUT_NAME",
    "QUANTIZED_OPS",
    "ExportBundle",
    "ExportResult",
    "LastStepLogits",
    "ParityResult",
    "QuantizeResult",
    "export_all",
    "export_onnx",
    "parity",
    "quantize_int8",
    "random_prefixes",
    "target_path",
    "to_fp16",
]
