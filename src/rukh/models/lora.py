"""LoRA written by hand: low-rank adapters over the frozen weights of a ``MoveDecoder``.

A full fine-tune of ``medium`` writes 115 M numbers and produces a 1.4 GB checkpoint, so a
project that wants five styles needs five copies of the same model. LoRA asks a different
question: instead of moving ``W``, learn a correction ``B A`` whose rank is small enough that the
two factors together are a rounding error next to ``W``. With ``r = 8`` on the query and value
projections of a 16-layer decoder that is 393 216 numbers -- 0.34 % of the model, 1.6 MB in
``float32`` -- and the base weights never change, so every adapter shares one copy of them.

Two details of *this* decoder shape the implementation.

The first is that attention uses one fused ``qkv`` projection (``layers.SelfAttention``), so
there is no ``q_proj`` module to wrap. Adapting "query and value" means adapting two contiguous
output ranges of a single ``nn.Linear``, which is why ``LoRALinear`` takes a tuple of output
slices and gives **each one its own ``A`` and ``B``**. Sharing one ``A`` between the query and
the value slice would be cheaper and would not be LoRA: the whole point is that each adapted
matrix gets its own subspace.

The second is that wrapping renames parameters: ``blocks.0.attn.qkv.weight`` becomes
``blocks.0.attn.qkv.base.weight``, which no existing checkpoint knows about. So the wrappers are
never what gets saved. ``save_adapter`` writes only ``A`` and ``B``, keyed by the path of the
module they adapt, and ``merge_lora`` folds the correction back into ``W`` and puts the plain
``nn.Linear`` back where it was, which is what the exporter and the publisher see.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pydantic import Field
from torch import Tensor, nn

from rukh.config import BaseConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "TARGETS",
    "LoraConfig",
    "LoRALinear",
    "adapter_params",
    "apply_lora",
    "load_adapter",
    "lora_modules",
    "merge_lora",
    "merged_state_dict",
    "save_adapter",
]

ADAPTER_FILE = "adapter.safetensors"
ADAPTER_CONFIG = "adapter_config.json"

# Logical target names and where they live in a block. The names are what a reader of the
# LoRA papers expects ("we adapt Wq and Wv"); the values say which module carries that matrix
# in this decoder and which output range of it belongs to the target. The fused projection
# writes query, key and value in that order, each `d_model` wide, so the ranges are given in
# units of `d_model` and resolved against the block at apply time.
TARGETS: dict[str, tuple[str, int, int]] = {
    "q": ("attn.qkv", 0, 1),
    "k": ("attn.qkv", 1, 2),
    "v": ("attn.qkv", 2, 3),
    "attn_out": ("attn.proj", 0, 1),
    "mlp_in": ("mlp.fc", 0, 1),
    "mlp_out": ("mlp.proj", 0, 1),
}


class LoraConfig(BaseConfig):
    """Rank, scaling and which matrices of each block get an adapter."""

    r: int = Field(default=8, ge=1)
    alpha: int = Field(default=16, ge=1)
    """The correction is scaled by ``alpha / r``, so raising the rank does not also raise how
    hard the adapter pushes. It is the convention of the paper and of ``peft``; keeping it means
    a rank sweep measures rank and not two things at once."""
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    targets: tuple[str, ...] = ("q", "v")

    def check(self) -> None:
        unknown = [name for name in self.targets if name not in TARGETS]
        if unknown:
            raise ValueError(f"unknown LoRA targets {unknown}; known: {sorted(TARGETS)}")
        if not self.targets:
            raise ValueError("targets must name at least one matrix")

    @property
    def scale(self) -> float:
        return self.alpha / self.r


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus one low-rank correction per adapted output slice.

    ``slices`` are half-open ``(start, stop)`` ranges over the output features. A plain linear
    gets one range covering everything; the fused ``qkv`` gets one range per adapted projection,
    which is how "query and value but not key" is expressed on a matrix that holds all three.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: int,
        dropout: float,
        slices: tuple[tuple[int, int], ...],
    ) -> None:
        super().__init__()
        if not slices:
            raise ValueError("a LoRALinear needs at least one output slice")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.slices = tuple(slices)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        in_features = base.in_features
        # Built where the weight they correct already lives. `apply_lora` runs *after* the model
        # has been moved to the device (the training loop loads the checkpoint first, and loading
        # needs the module names of a plain decoder), so a factor created on the CPU by default
        # would meet a CUDA activation on the first forward. Every unit test of this file runs on
        # the CPU, where the mistake is invisible.
        where = {"device": base.weight.device, "dtype": base.weight.dtype}
        # `A` is drawn from the same Kaiming-uniform the paper uses and `B` starts at exactly
        # zero, so the adapted model *is* the base model at step 0. Any other initialisation
        # would move the weights before a single gradient arrived.
        self.a = nn.ParameterList(
            nn.Parameter(torch.empty(r, in_features, **where)) for _ in range(len(self.slices))
        )
        self.b = nn.ParameterList(
            nn.Parameter(torch.zeros(stop - start, r, **where)) for start, stop in self.slices
        )
        for a in self.a:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        dropped = self.drop(x)
        delta = out.new_zeros(out.shape)
        for (start, stop), a, b in zip(self.slices, self.a, self.b, strict=True):
            delta[..., start:stop] = torch.nn.functional.linear(
                torch.nn.functional.linear(dropped, a), b
            )
        return out + self.scale * delta

    @torch.no_grad()
    def merged_weight(self) -> Tensor:
        """``W + (alpha / r) * B A`` with each correction written into its own output slice."""
        weight = self.base.weight.detach().clone()
        for (start, stop), a, b in zip(self.slices, self.a, self.b, strict=True):
            weight[start:stop] += self.scale * (b @ a).to(weight.dtype)
        return weight

    def extra_repr(self) -> str:
        return f"r={self.r}, scale={self.scale}, slices={self.slices}"


def _target_slices(
    cfg: LoraConfig, d_model: int, ff: int
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Group the configured targets by the module they live in, as output ranges of it."""
    by_module: dict[str, list[tuple[int, int]]] = {}
    for name in cfg.targets:
        module_name, lo, hi = TARGETS[name]
        width = ff if module_name == "mlp.fc" else d_model
        by_module.setdefault(module_name, []).append((lo * width, hi * width))
    return {name: tuple(sorted(ranges)) for name, ranges in by_module.items()}


def _get_module(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _set_module(root: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def apply_lora(model: nn.Module, cfg: LoraConfig) -> int:
    """Freeze ``model``, wrap the configured matrices of every block and return trainable params.

    Everything outside the adapters is frozen, including the embeddings and the final norm: a
    LoRA run that also trains the embedding table is not a LoRA run, and the count this returns
    would stop meaning what it says.
    """
    cfg.check()
    for param in model.parameters():
        param.requires_grad_(False)
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("model has no `blocks`; LoRA is applied per transformer block")
    d_model = model.cfg.d_model
    ff = getattr(model.cfg, "ff", 4 * d_model)
    plan = _target_slices(cfg, d_model, ff)
    for block in blocks:
        for module_name, slices in plan.items():
            base = _get_module(block, module_name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"{module_name} is a {type(base).__name__}, not nn.Linear")
            _set_module(block, module_name, LoRALinear(base, cfg.r, cfg.alpha, cfg.dropout, slices))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    """Every adapter in the model, by the dotted path of the matrix it adapts."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def adapter_params(model: nn.Module) -> Iterator[nn.Parameter]:
    """The parameters a LoRA run trains: the ``A`` and ``B`` of every adapter, nothing else."""
    for _, module in lora_modules(model):
        yield from module.a
        yield from module.b


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into the weight it adapts and put the plain ``nn.Linear`` back.

    After this the model is indistinguishable from one that was fine-tuned in full: same module
    names, same state dict, same ONNX graph. That is the property that lets an adapter be
    published as a 1.6 MB file and still be exportable as an ordinary model.
    """
    merged = 0
    for name, adapter in list(lora_modules(model)):
        base = adapter.base
        base.weight.copy_(adapter.merged_weight())
        base.weight.requires_grad_(True)
        if base.bias is not None:
            base.bias.requires_grad_(True)
        _set_module(model, name, base)
        merged += 1
    return merged


@torch.no_grad()
def merged_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """The state dict this model *would* have if every adapter were folded in, without folding.

    ``merge_lora`` mutates, which is what an export wants and what a training loop does not: a
    checkpoint written mid-run must not leave the model unable to take another step. So the
    folding is done on a copy of the numbers and the wrappers are left alone, and what comes out
    has the module names of a plain ``MoveDecoder`` -- which is what every loader, exporter and
    publisher in this project expects to read.
    """
    adapters = dict(lora_modules(model))
    merged: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        owner, _, leaf = name.rpartition(".")
        base_owner, _, base_leaf = owner.rpartition(".")
        if base_leaf == "base" and base_owner in adapters:
            value = adapters[base_owner].merged_weight() if leaf == "weight" else tensor
            merged[f"{base_owner}.{leaf}"] = value.detach().cpu()
            continue
        if any(name.startswith(f"{path}.{factor}.") for path in adapters for factor in ("a", "b")):
            continue  # the adapter's own factors; they live in the adapter file, not here
        merged[name] = tensor.detach().cpu()
    return merged


def save_adapter(model: nn.Module, path: Path | str, cfg: LoraConfig) -> Path:
    """Write ``A`` and ``B`` to safetensors, keyed by the module they adapt, plus the config.

    The keys are the paths the adapter *would* have in the base model
    (``blocks.0.attn.qkv.lora_A.0``), not the paths the wrappers create, so loading does not
    depend on how the wrapping happened to be spelled.
    """
    from safetensors.torch import save_file

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Tensor] = {}
    for name, adapter in lora_modules(model):
        for index, (a, b) in enumerate(zip(adapter.a, adapter.b, strict=True)):
            tensors[f"{name}.lora_A.{index}"] = a.detach().cpu().contiguous()
            tensors[f"{name}.lora_B.{index}"] = b.detach().cpu().contiguous()
    if not tensors:
        raise ValueError("model has no LoRA adapters to save")
    save_file(tensors, target.as_posix())
    config_path = target.with_name(ADAPTER_CONFIG)
    config_path.write_text(cfg.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


def load_adapter(model: nn.Module, path: Path | str) -> LoraConfig:
    """Apply the adapter stored next to ``path`` to a base model and load its weights."""
    from safetensors.torch import load_file

    target = Path(path)
    cfg = LoraConfig.model_validate_json(
        target.with_name(ADAPTER_CONFIG).read_text(encoding="utf-8")
    )
    apply_lora(model, cfg)
    tensors = load_file(target.as_posix())
    state: dict[str, Tensor] = {}
    for name, adapter in lora_modules(model):
        for index in range(len(adapter.a)):
            state[f"{name}.a.{index}"] = tensors[f"{name}.lora_A.{index}"]
            state[f"{name}.b.{index}"] = tensors[f"{name}.lora_B.{index}"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"adapter carries tensors the model has no place for: {unexpected}")
    return cfg


def adapter_size(model: nn.Module) -> tuple[int, int]:
    """``(adapter parameters, base parameters)``: the ratio the whole method exists for."""
    adapter = sum(p.numel() for p in adapter_params(model))
    total = sum(p.numel() for p in model.parameters())
    return adapter, total - adapter


def describe(model: nn.Module, cfg: LoraConfig) -> str:
    """One line for the run log: rank, targets and how little of the model is being trained."""
    adapter, base = adapter_size(model)
    share = 100.0 * adapter / (adapter + base)
    targets = ", ".join(cfg.targets)
    return (
        f"LoRA r={cfg.r} alpha={cfg.alpha} on [{targets}]: "
        f"{adapter:,} trainable of {adapter + base:,} ({share:.2f} %)"
    )


def adapter_json(cfg: LoraConfig, base_repo: str, extra: dict[str, object] | None = None) -> str:
    """The JSON that travels with a published adapter: what it is and what it mounts on."""
    payload: dict[str, object] = {"lora": cfg.model_dump(mode="json"), "base_model": base_repo}
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2) + "\n"
