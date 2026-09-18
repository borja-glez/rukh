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

``dynamic_seq`` is not taken on trust. The legacy tracer happily bakes the traced length into
the graph while still being asked for a dynamic axis, and the demo feeds a sequence that grows
by one token per move, so the exported file is **run** at two different lengths and the flag
reports what actually worked. The model's context (``block``) travels with the file as ONNX
metadata, so the browser knows the limit without being told separately.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from rukh import __version__
from rukh.models import MoveDecoder

log = logging.getLogger(__name__)

MODEL_NAME = "model.onnx"
INPUT_NAME = "idx"
OUTPUT_NAME = "logits"
BATCH_AXIS = "batch"
SEQUENCE_AXIS = "sequence"
DEFAULT_OPSET = 18
METADATA_PREFIX = "rukh_"


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
    block: int
    """The model's context: the exported graph must never be fed more than this many tokens."""
    dynamic_batch: bool
    dynamic_seq: bool
    """What the file really accepts, not what was asked for: verified by running it."""
    dynamic_seq_verified: bool = False
    """Whether ``dynamic_seq`` was checked by running the file (needs ``onnxruntime``)."""
    metadata: dict[str, str] = {}
    """``metadata_props`` written into the file, empty when ``onnx`` is not installed."""
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
    works = verify_dynamic_seq(path, seq_len, model.cfg.block)
    if dynamic_seq and works is False:
        raise ValueError(
            f"{path} was exported with a dynamic sequence axis but only runs at length "
            f"{seq_len}: the {exporter} exporter baked the length in. Re-export with "
            "dynamic_seq=False and pad the input, or fix the exporter."
        )
    really_dynamic = dynamic_seq if works is None else works
    metadata = write_metadata(
        path,
        {
            "block": model.cfg.block,
            "vocab_size": model.cfg.vocab_size,
            "seq_len": seq_len,
            "dynamic_batch": dynamic_batch,
            "dynamic_seq": really_dynamic,
            "exporter": exporter,
            "version": __version__,
        },
    )
    return ExportResult(
        path=path.as_posix(),
        exporter=exporter,
        opset=opset,
        seq_len=seq_len,
        block=model.cfg.block,
        dynamic_batch=dynamic_batch,
        dynamic_seq=really_dynamic,
        dynamic_seq_verified=works is not None,
        metadata=metadata,
        vocab_size=model.cfg.vocab_size,
        params=model.num_params(non_embedding=False),
        bytes=path.stat().st_size,
        warning=warning,
    )


def sequence_lengths(seq_len: int, block: int) -> list[int]:
    """Two lengths to run the exported file at: the traced one and a different, legal one."""
    other = seq_len + 1 if seq_len < block else max(1, seq_len - 1)
    return sorted({seq_len, other})


def verify_dynamic_seq(path: Path, seq_len: int, block: int) -> bool | None:
    """Run the file at two sequence lengths; None when ``onnxruntime`` is not installed."""
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime is not installed: the dynamic sequence axis was not verified")
        return None
    lengths = sequence_lengths(seq_len, block)
    if len(lengths) < 2:  # pragma: no cover - block of 1 is not a usable model
        return None
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for length in lengths:
        try:
            session.run(None, {INPUT_NAME: np.zeros((1, length), dtype=np.int64)})
        except Exception as exc:  # noqa: BLE001 - any refusal means the axis is not dynamic
            log.info("the exported graph refused a sequence of %d tokens: %s", length, exc)
            return False
    return True


def set_metadata(path: Path, entries: dict[str, str]) -> dict[str, str]:
    """Write ``metadata_props`` verbatim into a file; empty when ``onnx`` is not installed."""
    try:
        import onnx
    except ImportError:
        log.warning("onnx is not installed: the model metadata (block, vocab) was not written")
        return {}
    model = onnx.load(str(path))
    onnx.helper.set_model_props(model, entries)
    onnx.save(model, str(path))
    return entries


def write_metadata(path: Path, props: dict[str, Any]) -> dict[str, str]:
    """Write ``rukh_*`` metadata into the file; empty when ``onnx`` is not installed."""
    return set_metadata(
        path, {f"{METADATA_PREFIX}{key}": str(value) for key, value in props.items()}
    )


def read_metadata(path: Path) -> dict[str, str]:
    """The ``rukh_*`` metadata of an exported file (needs ``onnx``)."""
    import onnx

    model = onnx.load(str(path))
    return {entry.key: entry.value for entry in model.metadata_props}


def _load(ckpt: Path) -> MoveDecoder:
    from rukh.train import load_model

    model, _payload = load_model(ckpt)
    return model
