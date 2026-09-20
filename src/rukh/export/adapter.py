"""Exporting the decoder with its LoRA factors as **inputs of the graph**.

Merging an adapter and exporting the result is the obvious way to serve a style in the browser,
and it throws away the only thing LoRA bought: one style is 221 MB of ONNX for ``medium``, two
styles are 442 MB, and the 1.6 MB file stops meaning anything. So the graph written here keeps
the correction outside the weights. ``A`` and ``B`` arrive as two tensors on every call, stacked
over the layers (``rukh.models.lora.AdapterLayout``), and the browser changes style by uploading
1.6 MB instead of downloading a second model.

Three properties make the file honest, and all three are checked by the tests:

* **Zeros are the base model.** A graph that takes the factors as inputs and is fed zeros
  computes exactly what the plain export computes, to the last bit of float32. So the adaptable
  file *replaces* the ordinary one instead of being a second copy of it, and a demo with no
  style selected is not running a different model.
* **The factors are really inputs.** Feeding a different adapter changes the logits without
  touching the file. That is the whole claim, and it is one assertion away from being tested.
* **Parity with PyTorch.** The same adapter through ``apply_lora`` and through this graph has to
  give the same next move; ``rukh.export.parity`` measures it over real positions.

What it does not do is take an adapter of any shape. The factors travel as one tensor each, so
every block must adapt the same matrix with the same slice widths -- true of query, key and
value on the fused ``qkv``, false as soon as the MLP joins in, whose ``fc`` is four times wider.
That case still has ``merge_lora`` and an ordinary export; it just cannot be swapped at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rukh import __version__
from rukh.export.onnx import (
    DEFAULT_OPSET,
    INPUT_NAME,
    OUTPUT_NAME,
    ExportResult,
    _load,
    _write_graph,
    target_path,
    verify_dynamic_seq,
    write_metadata,
)
from rukh.models import MoveDecoder
from rukh.models.lora import (
    ADAPTER_CONFIG,
    AdapterLayout,
    LoraConfig,
    LoRALinear,
    adapter_layout,
    apply_lora,
    lora_modules,
    stack_adapter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ADAPTER_INPUTS",
    "A_INPUT",
    "B_INPUT",
    "WEB_ADAPTER_FILE",
    "WEB_ADAPTER_META",
    "AdaptableDecoder",
    "adaptable_inputs",
    "browser_lora",
    "build_adaptable",
    "export_adaptable_onnx",
    "write_web_adapter",
    "write_web_adapter_from_file",
    "zero_inputs",
]

A_INPUT = "lora_a"
B_INPUT = "lora_b"
ADAPTER_INPUTS = (A_INPUT, B_INPUT)

WEB_ADAPTER_FILE = "web/adapter.bin"
"""Where a published adapter keeps the copy the browser downloads: raw little-endian float32,
``A`` first and then ``B``, in the order the layers run."""

WEB_ADAPTER_META = "web/adapter.json"
"""Its shapes and provenance, because a flat buffer of floats says nothing about itself. The
sidecar is always the buffer's own name with a ``.json`` suffix, so a file written under another
name keeps its metadata next to it and not next to somebody else's."""


def browser_lora(cfg: LoraConfig) -> LoraConfig:
    """The same adapter shape with ``alpha = r``, which is the scaling the browser graph uses.

    The correction is ``(alpha / r) B A``, and that factor has to live in exactly one place. It
    could be the graph -- and then every adapter fed to it would have to have been trained with
    the same ``alpha`` and ``r``, with nothing in a flat buffer of floats to check it against --
    or it could be the file. It is the file: ``write_web_adapter`` multiplies ``B`` by the
    adapter's own scale before writing it, and the graph the browser runs multiplies by one. An
    adapter of any rank and any alpha is then correct in that graph without anyone matching two
    numbers by hand.
    """
    return cfg.model_copy(update={"alpha": cfg.r})


class _Slot:
    """Where one block's factors are put just before the block runs.

    Deliberately not an ``nn.Module`` attribute: ``torch.export`` warns that a tensor assigned to
    a module during tracing should have been a buffer, and a buffer is exactly what these must
    not be -- a buffer is baked into the file and the point is that they are not.
    """

    __slots__ = ("a", "b")

    a: Tensor
    b: Tensor


class SliceLoraFromInputs(nn.Module):
    """A frozen ``nn.Linear`` plus one low-rank correction per slice, taken from a ``_Slot``.

    The same arithmetic as ``LoRALinear.forward`` with one change that matters to ONNX: the
    correction is assembled with ``cat`` over the whole output width instead of being written
    into a slice of a zero tensor. Index assignment exports as ``ScatterND``, which onnxruntime
    web runs on the CPU even under WebGPU; a concatenation of matmuls and zeros stays on the
    accelerator.
    """

    def __init__(
        self, base: nn.Linear, slices: Sequence[tuple[int, int]], scale: float, slot: _Slot
    ) -> None:
        super().__init__()
        self.base = base
        self.slices = tuple(slices)
        self.scale = scale
        self.slot = slot

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        width = self.base.out_features
        pieces: list[Tensor] = []
        cursor = 0
        for index, (start, stop) in enumerate(self.slices):
            if start > cursor:
                pieces.append(out.new_zeros((*out.shape[:-1], start - cursor)))
            pieces.append(F.linear(F.linear(x, self.slot.a[index]), self.slot.b[index]))
            cursor = stop
        if cursor < width:
            pieces.append(out.new_zeros((*out.shape[:-1], width - cursor)))
        return out + self.scale * torch.cat(pieces, dim=-1)


class AdaptableDecoder(nn.Module):
    """``forward(idx, lora_a, lora_b)`` -> the last step's logits, with the adapter applied."""

    def __init__(self, model: MoveDecoder, slots: Sequence[_Slot]) -> None:
        super().__init__()
        self.model = model
        self.slots = list(slots)

    def forward(self, idx: Tensor, lora_a: Tensor, lora_b: Tensor) -> Tensor:
        for layer, slot in enumerate(self.slots):
            slot.a = lora_a[layer]
            slot.b = lora_b[layer]
        logits, _ = self.model(idx)
        return logits[:, -1, :]


def build_adaptable(model: MoveDecoder, cfg: LoraConfig) -> tuple[AdaptableDecoder, AdapterLayout]:
    """Wrap ``model`` so its adapted matrices read their factors from the forward's arguments.

    ``model`` may arrive plain or with an adapter already applied. Either way the wrappers that
    come out hold **no** factors of their own: whatever was loaded is returned by
    ``stack_adapter`` before the swap, so the caller can feed it back in and get the same model.
    """
    if not list(lora_modules(model)):
        apply_lora(model, cfg)
    layout = adapter_layout(model)
    slots: list[_Slot] = []
    for name, wrapper in list(lora_modules(model)):
        slot = _Slot()
        holder = SliceLoraFromInputs(wrapper.base, wrapper.slices, wrapper.scale, slot)
        parent, _, leaf = name.rpartition(".")
        setattr(_module(model, parent), leaf, holder)
        slots.append(slot)
    return AdaptableDecoder(model, slots).eval(), layout


def _module(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def zero_inputs(layout: AdapterLayout) -> dict[str, Any]:
    """The feed that makes the graph the base model: an adapter of all zeros."""
    import numpy as np

    return {
        A_INPUT: np.zeros(layout.shape_a, dtype=np.float32),
        B_INPUT: np.zeros(layout.shape_b, dtype=np.float32),
    }


def adaptable_inputs(model: nn.Module) -> dict[str, Any]:
    """The feed that applies the adapter currently loaded into ``model``, as numpy arrays."""
    a, b = stack_adapter(model)
    return {A_INPUT: a.numpy(), B_INPUT: b.numpy()}


def export_adaptable_onnx(
    ckpt: Path | MoveDecoder,
    out: Path,
    lora: LoraConfig | None = None,
    opset: int = DEFAULT_OPSET,
    dynamic_batch: bool = True,
    seq_len: int = 200,
    dynamic_seq: bool = True,
) -> ExportResult:
    """Export the next-move head with the LoRA factors as two extra inputs of the graph.

    The result is a drop-in replacement for ``export_onnx``'s file for any caller willing to
    feed the two tensors: with zeros it is the same model, bit for bit. ``lora`` describes the
    shape of the adapters the file will accept and defaults to the project's own recipe
    (``r = 8`` on the query and value ranges of the fused ``qkv``).
    """
    model = ckpt if isinstance(ckpt, MoveDecoder) else _load(Path(ckpt))
    model = model.eval()
    if seq_len > model.cfg.block:
        raise ValueError(f"seq_len={seq_len} is longer than the model's block {model.cfg.block}")
    wrapper, layout = build_adaptable(model, lora or LoraConfig())
    example = (
        torch.zeros((1, seq_len), dtype=torch.long),
        torch.zeros(layout.shape_a),
        torch.zeros(layout.shape_b),
    )
    path = target_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)

    inputs = (INPUT_NAME, *ADAPTER_INPUTS)
    exporter, warning = _write_graph(
        wrapper, example, path, opset, [OUTPUT_NAME], dynamic_batch, dynamic_seq, inputs
    )
    works = verify_dynamic_seq(path, seq_len, model.cfg.block, zero_inputs(layout))
    if dynamic_seq and works is False:
        raise ValueError(
            f"{path} was exported with a dynamic sequence axis but only runs at length "
            f"{seq_len}: the {exporter} exporter baked the length in."
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
            "adapter_inputs": ",".join(ADAPTER_INPUTS),
            "adapter_shape_a": ",".join(str(n) for n in layout.shape_a),
            "adapter_shape_b": ",".join(str(n) for n in layout.shape_b),
            "adapter_module": layout.module,
            "adapter_r": layout.r,
            "adapter_scale": layout.scale,
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


def _write_web(
    a: Tensor,
    b: Tensor,
    scale: float,
    module: str,
    r: int,
    out: Path | str,
    base_repo: str | None,
) -> Path:
    """The serialisation itself, shared by the two sources an adapter can come from."""
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(a.cpu().float().numpy().astype("<f4").tobytes())
        handle.write((b.cpu().float() * scale).numpy().astype("<f4").tobytes())
    meta = target.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "format": "rukh-lora-web-1",
                "dtype": "float32",
                "order": [A_INPUT, B_INPUT],
                "shape_a": list(a.shape),
                "shape_b": list(b.shape),
                "module": module,
                "r": r,
                "scale_applied": scale,
                "base_model": base_repo,
                "bytes": target.stat().st_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def write_web_adapter(model: nn.Module, out: Path | str, base_repo: str | None = None) -> Path:
    """Write the browser's copy of an adapter: one flat float32 buffer and its shapes.

    ``model`` is a decoder with the adapter already applied and loaded (``load_adapter``). What
    comes out is deliberately dull -- no safetensors header to parse in the worker, no shape
    inside the bytes -- because the page fetches it before it can do anything else and every
    dependency it needs is one more thing that can fail between the Hub and the board.

    ``B`` is written **scaled** (see ``browser_lora``), so the graph multiplies by one.
    """
    a, b = stack_adapter(model)
    layout = adapter_layout(model)
    return _write_web(a, b, layout.scale, layout.module, layout.r, out, base_repo)


def write_web_adapter_from_file(
    adapter: Path | str, out: Path | str, base_repo: str | None = None
) -> Path:
    """The same file, read straight from ``adapter.safetensors`` without a base model.

    Publishing an adapter should not need 440 MB of weights on disk to write 1.6 MB of floats,
    and the saved file already has everything: the keys carry the block number and the slice
    index, and ``adapter_config.json`` next to it carries the rank and the alpha. The order the
    tensors are stacked in is the order the blocks run, which is what the graph indexes by.
    """
    from safetensors.torch import load_file

    source = Path(adapter)
    cfg = LoraConfig.model_validate_json(
        (source.with_name(ADAPTER_CONFIG)).read_text(encoding="utf-8")
    )
    tensors = load_file(source.as_posix())
    a_keys = sorted(
        (key for key in tensors if ".lora_A." in key), key=lambda key: _adapter_sort_key(key)
    )
    if not a_keys:
        raise ValueError(f"{source} has no `lora_A` tensors")
    modules = sorted({key.split(".lora_A.")[0] for key in a_keys}, key=_adapter_sort_key)
    suffixes = {name.split(".", 2)[-1] for name in modules}
    if len(suffixes) != 1:
        raise ValueError(f"adapters on more than one matrix per block ({sorted(suffixes)})")
    per_layer = len(a_keys) // len(modules)
    a = torch.stack(
        [
            torch.stack([tensors[f"{name}.lora_A.{index}"] for index in range(per_layer)])
            for name in modules
        ]
    )
    b = torch.stack(
        [
            torch.stack([tensors[f"{name}.lora_B.{index}"] for index in range(per_layer)])
            for name in modules
        ]
    )
    return _write_web(a, b, cfg.scale, suffixes.pop(), cfg.r, out, base_repo)


def _adapter_sort_key(key: str) -> tuple[int, int]:
    """``blocks.10.attn.qkv.lora_A.1`` -> ``(10, 1)``: numeric, so 10 does not sort before 2."""
    parts = key.split(".")
    block = next((int(part) for part in parts if part.isdigit()), 0)
    index = int(parts[-1]) if parts[-1].isdigit() and ".lora_" in key else 0
    return block, index


def restore_lora_modules(model: MoveDecoder, cfg: LoraConfig) -> int:
    """Put ordinary ``LoRALinear`` wrappers back where ``build_adaptable`` left holders.

    Only the tests need this -- exporting is the end of a model's life in a process -- but they
    need it enough to keep it here rather than to reach into the internals from outside.
    """
    restored = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, SliceLoraFromInputs):
            continue
        wrapper = LoRALinear(module.base, cfg.r, cfg.alpha, cfg.dropout, module.slices)
        parent, _, leaf = name.rpartition(".")
        setattr(_module(model, parent), leaf, wrapper)
        restored += 1
    return restored
