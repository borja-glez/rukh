"""Checkpoints: what a run writes so it can be resumed, evaluated and published.

A payload holds only plain Python values and tensors, so it can be read back with
``torch.load(..., weights_only=True)``. Provenance (``vocab_hash``, ``data_manifest_sha``,
``git_sha``) is recorded on purpose: weights whose data or vocabulary cannot be identified are
not reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rukh.models import DecoderConfig, EncoderConfig, MoveDecoder, PositionEncoder
from rukh.models.heads import HeadWeights, MultiHead

log = logging.getLogger(__name__)

BEST_NAME = "best.pt"
RUN_STAMP = re.compile(r"^(?P<name>.+)-(?P<stamp>\d{8}-\d{6})$")
"""What ``unique_run_name`` appends to a run folder: ``medium-v4`` becomes
``medium-v4-20260919-174623``. The stamp sorts as text, so the newest run of a name is the last."""


def resolve_run(path: str | Path) -> Path:
    """The path as given when it exists; otherwise the same thing inside the newest stamped run.

    Configs and lessons name a checkpoint by its run: ``checkpoints/medium-v4/best.pt``. That is
    the spelling ``rukh pull`` writes, and it is not the one a reader who trained the run has,
    because ``unique_run_name`` stamps every folder (``medium-v4-20260919-174623``). Both mean
    "the run called medium-v4", so when the stable folder is missing this looks for the stamped
    ones beside it and takes the newest **by name**, never by modification time: copying a
    folder must not change which run a config points at. A folder spec (``checkpoints/lora-e4``)
    resolves the same way. Anything with no candidate comes back unchanged, and whoever opens it
    reports the missing file with the path the user wrote.
    """
    path = Path(path)
    if path.exists():
        return path
    is_file = bool(path.suffix)
    run_dir = path.parent if is_file else path
    parent, name = run_dir.parent, run_dir.name
    if not parent.is_dir():
        return path
    stamped = sorted(
        candidate
        for candidate in parent.iterdir()
        if candidate.is_dir()
        and (match := RUN_STAMP.match(candidate.name)) is not None
        and match.group("name") == name
        and (not is_file or (candidate / path.name).is_file())
    )
    if not stamped:
        return path
    found = stamped[-1] / path.name if is_file else stamped[-1]
    log.info("%s is not there; using the newest run of that name, %s", path, found)
    return found


CURVE_KEY = "label_curve"
"""Where ``rukh.train.heads.label_curve`` writes its points and ``rukh eval encoder`` reads them.

The curve is the lesson of M3 ("how many labels does this actually take?") and a
``docs/acceptance.md``
deliverable, so it travels **inside** the checkpoint: a number that only ever existed in a log
line is a number nobody can put in a table."""
TIED_HEAD = "lm_head.weight"
"""Tied to ``tokens.weight``; a published state dict leaves it out and it is re-tied on load."""
TIED_HEADS = frozenset({TIED_HEAD, "mlm_head.weight", "encoder.mlm_head.weight"})
"""Every tied head in the project: the decoder's and the encoder's masked-move head.

The last name is the same head seen from a ``MultiHead``, which holds the encoder as a child."""
TIED_SOURCES: dict[str, str] = {
    TIED_HEAD: "tokens.weight",
    "mlm_head.weight": "tokens.weight",
    "encoder.mlm_head.weight": "encoder.tokens.weight",
}
"""The embedding each tied head shares its storage with, when the tie is switched on."""


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
    state: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    """Write one checkpoint atomically (temporary file plus replace) and return its path.

    ``run_id`` is the MLflow run that wrote it, so ``--resume`` can carry on logging into the
    same run and the loss curve stays one line instead of two.

    ``state`` replaces ``model.state_dict()``. A LoRA run passes the merged weights, so what
    lands on disk is an ordinary decoder checkpoint that every loader, exporter and publisher
    already reads, instead of a wrapped model only this repository could open.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = state if state is not None else model.state_dict()
    payload: dict[str, Any] = {
        "step": int(step),
        "model_state": {k: v.detach().cpu() for k, v in source.items()},
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
    """Read a checkpoint written by ``save_checkpoint`` (see ``resolve_run`` for the path)."""
    payload = torch.load(resolve_run(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"{path} is not a rukh checkpoint")
    return payload


def attach_payload(path: Path, **extra: Any) -> Path:
    """Add plain values to an existing checkpoint, rewriting it atomically.

    Used for facts a run only knows once it is over — the label-count curve is the whole of it —
    so they do not need a second file next to the weights that can be lost or go stale.
    """
    path = Path(path)
    payload = load_checkpoint(path)
    payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_state(model: nn.Module, state: Mapping[str, Any]) -> None:
    """Load weights, accepting a state dict whose tied head was left out when it was saved.

    ``MoveDecoder`` ties ``lm_head.weight`` to ``tokens.weight`` in its constructor, so the tied
    head is already correct once the embedding is loaded: a state dict without it (the one
    ``rukh publish`` writes, because two names for one tensor is what makes ``safetensors``
    refuse the file) loads cleanly and nothing else may be missing.
    """
    missing, unexpected = model.load_state_dict(dict(state), strict=False)
    absent = [name for name in missing if name not in TIED_HEADS]
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


def load_encoder(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[PositionEncoder, dict[str, Any]]:
    """Rebuild the encoder a checkpoint describes, in eval mode, plus the whole payload."""
    payload = load_checkpoint(path, map_location=map_location)
    model = PositionEncoder(EncoderConfig.model_validate(payload["model_cfg"]))
    load_state(model, payload["model_state"])
    return model.eval(), payload


def build_heads(payload: Mapping[str, Any]) -> MultiHead:
    """A ``MultiHead`` shaped by a payload, with no weights loaded yet."""
    run_cfg = payload.get("cfg") or {}
    encoder = PositionEncoder(EncoderConfig.model_validate(payload["model_cfg"]))
    pooling = run_cfg.get("pooling") or "mean"
    if pooling not in ("cls", "mean"):
        raise ValueError(f"unknown pooling {pooling!r} in the checkpoint's config")
    return MultiHead(encoder, HeadWeights.model_validate(run_cfg.get("weights") or {}), pooling)


def load_heads(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[MultiHead, dict[str, Any]]:
    """Rebuild a fine-tuned ``MultiHead`` (encoder plus the three heads) from a checkpoint.

    ``rukh.train.heads`` saves the whole ``MultiHead`` state under the *encoder's* ``model_cfg``,
    because the heads have no shape of their own beyond ``d_model``; the pooling and the head
    weights come from the run's config, which travels in the same payload.
    """
    payload = load_checkpoint(path, map_location=map_location)
    model = build_heads(payload)
    load_state(model, payload["model_state"])
    return model.eval(), payload


def checkpoint_kind(payload: Mapping[str, Any]) -> str:
    """``"encoder"`` or ``"decoder"``: what kind of model a checkpoint describes.

    Read off the weights rather than off a flag nobody wrote: a ``MultiHead`` keeps the encoder
    as a child (``encoder.*``), a bare ``PositionEncoder`` is identified by the ``input`` field
    only its config has, and everything else is the decoder.
    """
    state = payload.get("model_state") or {}
    if any(str(name).startswith("encoder.") for name in state):
        return "encoder"
    return "encoder" if "input" in (payload.get("model_cfg") or {}) else "decoder"


def load_any(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any], str]:
    """Rebuild whatever model a checkpoint describes: ``(model, payload, kind)``.

    ``rukh export`` and ``rukh publish`` take a checkpoint and have to work out what is in it;
    this is the one place that decides, so the two commands can never disagree.
    """
    payload = load_checkpoint(path, map_location=map_location)
    kind = checkpoint_kind(payload)
    if kind == "decoder":
        model: nn.Module = MoveDecoder(DecoderConfig.model_validate(payload["model_cfg"]))
    elif any(str(name).startswith("encoder.") for name in payload["model_state"]):
        model = build_heads(payload)
    else:
        model = PositionEncoder(EncoderConfig.model_validate(payload["model_cfg"]))
    load_state(model, payload["model_state"])
    return model.eval(), payload, kind


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
