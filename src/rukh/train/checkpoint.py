"""Checkpoints: what a run writes so it can be resumed, evaluated and published.

A payload holds only plain Python values and tensors, so it can be read back with
``torch.load(..., weights_only=True)``. Provenance (``vocab_hash``, ``data_manifest_sha``,
``git_sha``) is recorded on purpose: weights whose data or vocabulary cannot be identified are
not reproducible.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rukh.models import DecoderConfig, MoveDecoder

BEST_NAME = "best.pt"
TIED_HEAD = "lm_head.weight"
"""Tied to ``tokens.weight``; a published state dict leaves it out and it is re-tied on load."""


def step_name(step: int) -> str:
    """File name of the checkpoint written after ``step`` optimizer steps."""
    return f"step-{step}.pt"


def read_vocab_hash(tokens_dir: Path) -> str | None:
    """``vocab_hash`` from a pack ``meta.json``, or None when it cannot be read."""
    try:
        meta = json.loads((Path(tokens_dir) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = meta.get("vocab_hash")
    return str(value) if isinstance(value, str) else None


def read_manifest_sha(manifest: Path) -> str | None:
    """SHA-256 of a data manifest, or None when the file is absent or unreadable.

    Training must never depend on ``data/`` being present: a checkpoint trained from a pack
    copied elsewhere simply records ``None``.
    """
    try:
        payload = Path(manifest).read_bytes()
    except OSError:
        return None
    return hashlib.sha256(payload).hexdigest()


def rng_state() -> dict[str, Any]:
    """Snapshot of the Python, NumPy and torch CPU generators."""
    name, keys, pos, has_gauss, cached = np.random.get_state(legacy=True)
    return {
        "python": json.dumps(random.getstate()),
        "numpy": json.dumps(
            [name, np.asarray(keys).tolist(), int(pos), int(has_gauss), float(cached)]
        ),
        "torch": torch.get_rng_state(),
    }


def set_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a snapshot taken by ``rng_state`` (missing entries are skipped)."""
    python = state.get("python")
    if isinstance(python, str):
        version, keys, gauss = json.loads(python)
        random.setstate((version, tuple(keys), gauss))
    numpy = state.get("numpy")
    if isinstance(numpy, str):
        name, keys, pos, has_gauss, cached = json.loads(numpy)
        np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.to(torch.uint8).cpu())


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    vocab_hash: str | None = None,
    data_manifest_sha: str | None = None,
    git_sha: str | None = None,
    best_val: float | None = None,
    run_id: str | None = None,
) -> Path:
    """Write one checkpoint atomically (temporary file plus replace) and return its path.

    ``run_id`` is the MLflow run that wrote it, so ``--resume`` can carry on logging into the
    same run and the loss curve stays one line instead of two.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": int(step),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "opt_state": optimizer.state_dict() if optimizer is not None else None,
        "cfg": dict(cfg),
        "model_cfg": dict(model_cfg),
        "vocab_hash": vocab_hash,
        "data_manifest_sha": data_manifest_sha,
        "git_sha": git_sha,
        "best_val": best_val,
        "run_id": run_id,
        "rng": rng_state(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a checkpoint written by ``save_checkpoint``."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"{path} is not a rukh checkpoint")
    return payload


def load_state(model: nn.Module, state: Mapping[str, Any]) -> None:
    """Load weights, accepting a state dict whose tied head was left out when it was saved.

    ``MoveDecoder`` ties ``lm_head.weight`` to ``tokens.weight`` in its constructor, so the tied
    head is already correct once the embedding is loaded: a state dict without it (the one
    ``rukh publish`` writes, because two names for one tensor is what makes ``safetensors``
    refuse the file) loads cleanly and nothing else may be missing.
    """
    missing, unexpected = model.load_state_dict(dict(state), strict=False)
    absent = [name for name in missing if name != TIED_HEAD]
    if absent or unexpected:
        raise ValueError(
            f"the weights do not match the model: missing {absent}, unexpected {list(unexpected)}"
        )


def load_model(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[MoveDecoder, dict[str, Any]]:
    """Rebuild the decoder a checkpoint describes, in eval mode, plus the whole payload."""
    payload = load_checkpoint(path, map_location=map_location)
    model = MoveDecoder(DecoderConfig.model_validate(payload["model_cfg"]))
    load_state(model, payload["model_state"])
    return model.eval(), payload


def restore(
    payload: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    with_rng: bool = True,
) -> int:
    """Load weights (and optionally the optimizer and RNG) and return the step reached."""
    load_state(model, payload["model_state"])
    opt_state = payload.get("opt_state")
    if optimizer is not None and opt_state is not None:
        optimizer.load_state_dict(opt_state)
    rng = payload.get("rng")
    if with_rng and isinstance(rng, Mapping):
        set_rng_state(rng)
    return int(payload["step"])
