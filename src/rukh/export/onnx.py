"""Exporting a ``MoveDecoder`` (and the encoder's heads) to ONNX for the browser.

The demo only ever needs the distribution over the *next* move, so the exported graph is not the
model itself but a wrapper whose forward returns the last step only: ``(B, V)`` instead of
``(B, T, V)``, which divides the output tensor by the context length (200) and saves the browser
from slicing a megabyte of logits per move.

``torch.onnx.export`` is tried with the dynamo exporter first (the default since torch 2.9 and
what ``docs/spec/02`` asks for) and falls back to the legacy TorchScript tracer with a warning
when dynamo is unavailable or fails; which path produced the file is recorded in the metadata,
because the two exporters do not emit the same graph and a parity check is only meaningful when
it is known which one ran.

The encoder travels the same road with a different wrapper: ``EncoderHeads`` returns the two
outputs the demo needs (``value`` and ``blunder``) and nothing else, the ``squares`` scheme has
no dynamic sequence axis at all (it is always 69 tokens) and the metadata carries ``rukh_kind``
and ``rukh_heads`` so a file can say what it is. Everything else — the exporter, the fallback,
the verification, the metadata, the sidecar — is the same code: the decoder's path is untouched.

``dynamic_seq`` is not taken on trust. The legacy tracer happily bakes the traced length into
the graph while still being asked for a dynamic axis, and the demo feeds a sequence that grows
by one token per move, so the exported file is **run** at two different lengths and the flag
reports what actually worked. The model's context (``block``) travels with the file as ONNX
metadata, so the browser knows the limit without being told separately.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from rukh import __version__
from rukh.models import MoveDecoder
from rukh.models.heads import MultiHead

log = logging.getLogger(__name__)

MODEL_NAME = "model.onnx"
INPUT_NAME = "idx"
OUTPUT_NAME = "logits"
VALUE_OUTPUT = "value"
BLUNDER_OUTPUT = "blunder"
ENCODER_OUTPUTS = (VALUE_OUTPUT, BLUNDER_OUTPUT)
"""What the encoder graph returns: the evaluation bar and the blunder alert of the demo."""
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


class EncoderHeads(nn.Module):
    """Wraps a fine-tuned encoder so that ``forward(idx)`` returns ``(value, blunder)``.

    The third head (``result``) is left out on purpose: the demo draws an evaluation bar and a
    blunder alert, and every tensor the graph returns is one the browser has to read back over
    the worker boundary. ``blunder`` comes out as a probability rather than a logit, so the page
    compares it against 0.5 instead of carrying a sigmoid of its own.
    """

    def __init__(self, model: MultiHead) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        pooled = self.model.pooled(idx)
        return self.model.value(pooled), torch.sigmoid(self.model.blunder(pooled))


class ExportResult(BaseModel):
    """What was written and how."""

    model_config = ConfigDict(extra="forbid")

    path: str
    kind: Literal["decoder", "encoder"] = "decoder"
    outputs: list[str] = [OUTPUT_NAME]
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


def _dynamic_axes(
    dynamic_batch: bool, dynamic_seq: bool, outputs: Sequence[str] = (OUTPUT_NAME,)
) -> dict[str, dict[int, str]] | None:
    shapes = _dynamic_shapes(dynamic_batch, dynamic_seq)
    if shapes is None:
        return None
    axes = dict(shapes)
    if dynamic_batch:
        for name in outputs:
            axes[name] = {0: BATCH_AXIS}
    return axes


@contextlib.contextmanager
def _utf8_console() -> Iterator[None]:
    """Let the exporter print its progress ticks on a legacy Windows code page.

    ``torch.onnx.export(dynamo=True)`` writes check marks to stdout. On Windows the console
    is cp1252 by default, so that raises ``UnicodeEncodeError`` inside the exporter and the
    whole export fails for a reason that has nothing to do with the model. Reconfiguring the
    streams to replace unencodable characters keeps the modern exporter usable; without it
    every Windows export silently falls back to the deprecated TorchScript one, which in turn
    produces an fp16 graph that onnxruntime refuses to load.
    """
    streams = [s for s in (sys.stdout, sys.stderr) if hasattr(s, "reconfigure")]
    previous = [(s, s.encoding, s.errors) for s in streams]
    for stream in streams:
        with contextlib.suppress(Exception):
            stream.reconfigure(errors="replace")
    try:
        yield
    finally:
        for stream, encoding, errors in previous:
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding=encoding, errors=errors)


def _write_graph(
    wrapper: nn.Module,
    example: Tensor,
    path: Path,
    opset: int,
    outputs: Sequence[str],
    dynamic_batch: bool,
    dynamic_seq: bool,
) -> tuple[Literal["dynamo", "legacy"], str | None]:
    """Write one ONNX file with the modern exporter, falling back to the legacy one.

    Both graphs of the project (the decoder's next move and the encoder's two heads) are written
    here so that the fallback, the console workaround and the axis declarations exist once.
    """
    common: dict[str, Any] = {
        "input_names": [INPUT_NAME],
        "output_names": list(outputs),
        "opset_version": opset,
    }
    try:
        with torch.no_grad(), _utf8_console():
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
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (example,),
                str(path),
                dynamo=False,
                dynamic_axes=_dynamic_axes(dynamic_batch, dynamic_seq, outputs),
                **common,
            )
        return "legacy", warning
    return "dynamo", None


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

    exporter, warning = _write_graph(
        wrapper, example, path, opset, [OUTPUT_NAME], dynamic_batch, dynamic_seq
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


def export_encoder_onnx(
    ckpt: Path | MultiHead,
    out: Path,
    opset: int = DEFAULT_OPSET,
    dynamic_batch: bool = True,
) -> ExportResult:
    """Export the ``value`` and ``blunder`` heads of a fine-tuned encoder to ONNX.

    The length of the input is the scheme's, not the caller's: ``squares`` is always 69 tokens
    and there is no sequence axis to make dynamic, while ``moves`` reads a growing game exactly
    as the decoder does. The example the exporter traces is a sequence of real tokens rather
    than zeros, because ``<pad>`` everywhere is a position the encoder legitimately refuses.
    """
    model = ckpt if isinstance(ckpt, MultiHead) else _load_heads(Path(ckpt))
    model = model.eval()
    cfg = model.encoder.cfg
    seq_len = cfg.seq
    dynamic_seq = cfg.input == "moves"
    wrapper = EncoderHeads(model).eval()
    # Two rows, not one, and no zeros. `torch.export` specialises a dimension whose example is
    # 1 (the 0/1 specialisation), so a batch of one is baked into the graph and the demo's
    # second position fails inside a reshape; tracing at two keeps the axis dynamic and the file
    # still accepts a batch of one. `<pad>` everywhere would be an entirely masked row, which
    # the encoder refuses on purpose, so the example is made of real token ids.
    example = torch.ones((2, seq_len), dtype=torch.long)
    path = target_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)

    exporter, warning = _write_graph(
        wrapper, example, path, opset, list(ENCODER_OUTPUTS), dynamic_batch, dynamic_seq
    )
    works = verify_dynamic_seq(path, seq_len, cfg.seq) if dynamic_seq else None
    if dynamic_seq and works is False:
        raise ValueError(
            f"{path} was exported with a dynamic sequence axis but only runs at length "
            f"{seq_len}: the {exporter} exporter baked the length in."
        )
    really_dynamic = dynamic_seq if works is None else works
    clear_value_info(path)
    metadata = write_metadata(
        path,
        {
            "kind": "encoder",
            "heads": ",".join(ENCODER_OUTPUTS),
            "input": cfg.input,
            "pooling": model.pooling,
            "blunder": "probability",
            "block": seq_len,
            "vocab_size": cfg.tokens,
            "seq_len": seq_len,
            "dynamic_batch": dynamic_batch,
            "dynamic_seq": really_dynamic,
            "exporter": exporter,
            "version": __version__,
        },
    )
    return ExportResult(
        path=path.as_posix(),
        kind="encoder",
        outputs=list(ENCODER_OUTPUTS),
        exporter=exporter,
        opset=opset,
        seq_len=seq_len,
        block=seq_len,
        dynamic_batch=dynamic_batch,
        dynamic_seq=really_dynamic,
        dynamic_seq_verified=works is not None,
        metadata=metadata,
        vocab_size=cfg.tokens,
        params=sum(parameter.numel() for parameter in model.parameters()),
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
    # The dynamo exporter writes the weights next to the graph as `<name>.onnx.data` and points
    # the initializers at it. `onnx.load` above pulled them into memory and `onnx.save` wrote
    # them back inline, so the sidecar is now dead weight that would ship to the browser (or be
    # published to the Hub) without ever being read. Drop it, but only once the file really is
    # self-contained.
    sidecar = path.with_suffix(path.suffix + ".data")
    if sidecar.is_file() and not any(
        tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer
    ):
        sidecar.unlink()
        log.debug("removed the external-data sidecar %s", sidecar.name)
    return entries


def write_metadata(path: Path, props: dict[str, Any]) -> dict[str, str]:
    """Write ``rukh_*`` metadata into the file; empty when ``onnx`` is not installed."""
    return set_metadata(
        path, {f"{METADATA_PREFIX}{key}": str(value) for key, value in props.items()}
    )


def clear_value_info(path: Path) -> int:
    """Drop the graph's annotations for intermediate values; returns how many were removed.

    The modern exporter records a shape for every intermediate tensor, and some of those are the
    shape of the traced example rather than of the dynamic graph. Nothing runs them:
    onnxruntime executes the file regardless. But ``quantize_dynamic`` re-runs shape inference
    in strict mode first and refuses a file whose annotations disagree with what it infers, so
    the int8 build of the encoder dies on an annotation instead of on a weight. They are
    optional by the specification, so the encoder's graph goes out without them.
    """
    try:
        import onnx
    except ImportError:  # pragma: no cover - onnx is a hard dependency of the exporter
        return 0
    model = onnx.load(str(path))
    removed = len(model.graph.value_info)
    del model.graph.value_info[:]
    onnx.save(model, str(path))
    return removed


def read_metadata(path: Path) -> dict[str, str]:
    """The ``rukh_*`` metadata of an exported file (needs ``onnx``)."""
    import onnx

    model = onnx.load(str(path))
    return {entry.key: entry.value for entry in model.metadata_props}


def _load(ckpt: Path) -> MoveDecoder:
    from rukh.train import load_model

    model, _payload = load_model(ckpt)
    return model


def _load_heads(ckpt: Path) -> MultiHead:
    from rukh.train import load_heads

    model, _payload = load_heads(ckpt)
    return model
