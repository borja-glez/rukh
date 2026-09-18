"""Exporting a ``MoveDecoder`` to ONNX for the browser.

The demo only ever needs the distribution over the *next* move, so the exported graph is not the
model itself but a wrapper whose forward returns the last step only: ``(B, V)`` instead of
``(B, T, V)``, which divides the output tensor by the context length (200) and saves the browser
from slicing a megabyte of logits per move.

``torch.onnx.export`` is tried with the dynamo exporter first (the default since torch 2.9 and
what ``docs/spec/02`` asks for) and falls back to the legacy TorchScript tracer with a warning
when dynamo is unavailable or fails; which path produced the file is recorded in the metadata,
because the two exporters do not emit the same graph and a parity check is only meaningful when
it is known which one ran.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from rukh.models import MoveDecoder

log = logging.getLogger(__name__)

MODEL_NAME = "model.onnx"
INPUT_NAME = "idx"
OUTPUT_NAME = "logits"
BATCH_AXIS = "batch"
SEQUENCE_AXIS = "sequence"
DEFAULT_OPSET = 18


class LastStepLogits(nn.Module):
    """Wraps a decoder so that ``forward(idx)`` returns only the last step's logits."""

    def __init__(self, model: MoveDecoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx: Tensor) -> Tensor:
        logits, _ = self.model(idx)
        return logits[:, -1, :]


class ExportResult(BaseModel):
    """What was written and how."""

    model_config = ConfigDict(extra="forbid")

    path: str
    exporter: Literal["dynamo", "legacy"]
    opset: int
    seq_len: int
    dynamic_batch: bool
    dynamic_seq: bool
    vocab_size: int
    params: int
    bytes: int
    warning: str | None = None


def target_path(out: Path) -> Path:
    """``out`` as a file: a directory gets ``model.onnx`` inside it."""
    out = Path(out)
    return out if out.suffix == ".onnx" else out / MODEL_NAME


def _dynamic_shapes(dynamic_batch: bool, dynamic_seq: bool) -> dict[str, dict[int, str]] | None:
    axes: dict[int, str] = {}
    if dynamic_batch:
        axes[0] = BATCH_AXIS
    if dynamic_seq:
        axes[1] = SEQUENCE_AXIS
    return {INPUT_NAME: axes} if axes else None


def _dynamic_axes(dynamic_batch: bool, dynamic_seq: bool) -> dict[str, dict[int, str]] | None:
    shapes = _dynamic_shapes(dynamic_batch, dynamic_seq)
    if shapes is None:
        return None
    axes = dict(shapes)
    if dynamic_batch:
        axes[OUTPUT_NAME] = {0: BATCH_AXIS}
    return axes


def export_onnx(
    ckpt: Path | MoveDecoder,
    out: Path,
    opset: int = DEFAULT_OPSET,
    dynamic_batch: bool = True,
    seq_len: int = 200,
    dynamic_seq: bool = True,
) -> ExportResult:
    """Export the next-move head of a checkpoint to ONNX and report which exporter ran.

    ``ckpt`` may be a checkpoint path or an already-built model (the tests use the latter).
    ``seq_len`` is the length of the example the exporter traces; with ``dynamic_seq`` the graph
    still accepts any length up to the model's block, which is what the demo feeds it as a game
    grows move by move.
    """
    model = ckpt if isinstance(ckpt, MoveDecoder) else _load(Path(ckpt))
    model = model.eval()
    if seq_len > model.cfg.block:
        raise ValueError(f"seq_len={seq_len} is longer than the model's block {model.cfg.block}")
    wrapper = LastStepLogits(model).eval()
    example = torch.zeros((1, seq_len), dtype=torch.long)
    path = target_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)

    common: dict[str, Any] = {
        "input_names": [INPUT_NAME],
        "output_names": [OUTPUT_NAME],
        "opset_version": opset,
    }
    warning: str | None = None
    exporter: Literal["dynamo", "legacy"] = "dynamo"
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (example,),
                str(path),
                dynamo=True,
                dynamic_shapes=_dynamic_shapes(dynamic_batch, dynamic_seq),
                **common,
            )
    except Exception as exc:  # noqa: BLE001 - any exporter failure must fall back, not stop
        warning = f"the dynamo exporter failed ({type(exc).__name__}: {exc}); used the legacy one"
        log.warning("%s", warning)
        exporter = "legacy"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (example,),
                str(path),
                dynamo=False,
                dynamic_axes=_dynamic_axes(dynamic_batch, dynamic_seq),
                **common,
            )
    return ExportResult(
        path=path.as_posix(),
        exporter=exporter,
        opset=opset,
        seq_len=seq_len,
        dynamic_batch=dynamic_batch,
        dynamic_seq=dynamic_seq,
        vocab_size=model.cfg.vocab_size,
        params=model.num_params(non_embedding=False),
        bytes=path.stat().st_size,
        warning=warning,
    )


def _load(ckpt: Path) -> MoveDecoder:
    from rukh.train import load_model

    model, _payload = load_model(ckpt)
    return model
