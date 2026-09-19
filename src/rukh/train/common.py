"""What every training run in this project does the same way, whatever it is training.

``loop.py`` (next-move prediction) and ``mmm.py`` (masked move modeling) differ in exactly one
place: how a batch becomes a loss. Everything around that step — the optimizer's parameter
groups, the learning-rate schedule, resuming from a checkpoint, writing one, choosing the
device, probing ``torch.compile`` and sending metrics to MLflow — is identical, so it lives here
and both loops call it. Duplicating it would mean two recipes drifting apart, and a run is only
comparable with another one when the recipe is the same.

``RunConfig`` is the shared half of both YAML configs; each loop's config adds what only it
needs (the decoder preset, or the masking parameters).
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from rukh import paths
from rukh.config import BaseConfig
from rukh.train.checkpoint import load_checkpoint, restore, save_checkpoint
from rukh.train.schedule import lr_at

log = logging.getLogger(__name__)

Batch = tuple[torch.Tensor, torch.Tensor]


class RunConfig(BaseConfig):
    """The half of a training config that does not depend on what is being trained."""

    tokens_dir: str = "data/tokens/uci"
    block: int = 200
    batch_size: int = 64
    grad_accum: int = 4  # 256 effective sequences
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup: int = 1000
    max_steps: int = 20000
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    precision: Literal["bf16", "fp32"] = "bf16"
    compile: bool = True
    eval_every: int = 500
    eval_batches: int = 50
    ckpt_every: int = 1000
    out_dir: str = "checkpoints"
    seed: int = 42
    run_name: str | None = None
    unique_run_name: bool = True
    """Append a timestamp to ``run_name``: a second run must not overwrite the ``step-*.pt``
    series the ``TrainingReplay`` of the course reads."""
    workers: int = 0  # DataLoader workers; 0 keeps everything in the main process
    log_every: int = 10  # optimizer steps between training metrics

    @property
    def default_name(self) -> str:
        """Run name when the YAML gives none; each loop names its runs after what it trains."""
        return "run"

    def check(self) -> None:
        """Refuse a config no loop could honour."""
        if min(self.max_steps, self.grad_accum, self.batch_size) < 1:
            raise ValueError("max_steps, grad_accum and batch_size must be positive")


def pick_device() -> str:
    """``RUKH_DEVICE`` if set, else CUDA when available, else CPU."""
    env = os.environ.get("RUKH_DEVICE")
    if env:
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Decoupled weight decay on matrices only: norms and biases are left alone."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def set_lr(optimizer: torch.optim.Optimizer, step: int, cfg: RunConfig) -> float:
    """Apply the scheduled learning rate of ``step`` to every group and return it."""
    lr = lr_at(step, cfg)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def forever(loader: DataLoader[Batch]) -> Iterator[Batch]:
    """Repeat a loader for as many steps as the schedule asks for."""
    while True:
        yield from loader


def autocast_for(cfg: RunConfig, where: torch.device) -> tuple[Any, bool]:
    """``(context, enabled)``: bf16 autocast on CUDA when asked, a no-op context otherwise."""
    use_bf16 = cfg.precision == "bf16" and where.type == "cuda"
    if not use_bf16:
        return nullcontext(), False
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16), True


def maybe_compile(
    model: nn.Module,
    enabled: bool,
    sample: torch.Tensor | None = None,
    probe: Callable[[nn.Module], None] | None = None,
) -> nn.Module:
    """``torch.compile`` the model when asked; a failure is a warning, never a stopped run.

    ``torch.compile`` is lazy: on Windows without MSVC it only fails when the first forward
    reaches Inductor. ``sample`` (one batch of the training shape, used as input and target by
    the decoder's ``forward``) or a ``probe`` that runs one step itself forces that compilation
    here, where it can still fall back to eager, and Dynamo's own error suppression covers any
    later recompilation for a different shape.
    """
    if not enabled:
        return model
    try:
        import torch._dynamo as dynamo

        dynamo.config.suppress_errors = True
        compiled = torch.compile(model)
        if probe is not None:
            probe(compiled)
            model.zero_grad(set_to_none=True)
        elif sample is not None:
            _, loss = compiled(sample, sample)
            if loss is not None:
                loss.backward()
            model.zero_grad(set_to_none=True)
        return compiled
    except Exception as exc:  # noqa: BLE001 - compilation backends fail in many ways
        log.warning("torch.compile is unavailable, training eagerly: %s", exc)
        model.zero_grad(set_to_none=True)
        return model


def run_dir(cfg: RunConfig, resume: Path | None = None) -> Path:
    """Where this run writes: the resumed run's folder, or ``out_dir/<run name>``.

    The name carries a timestamp unless ``unique_run_name`` is off, because two runs of the same
    config would otherwise write the same ``step-*.pt`` files and the second would quietly
    overwrite the checkpoint series of the first.
    """
    if resume is not None:
        return Path(resume).resolve().parent
    name = cfg.run_name or cfg.default_name
    if cfg.unique_run_name:
        name = f"{name}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    return paths.resolve(cfg.out_dir) / name


def skip_batches(batches: Iterator[Batch], count: int) -> int:
    """Wind the batch stream forward ``count`` batches and return how many were skipped.

    A resumed run must not start again at the first window of the first epoch: it would train
    twice on the same games while the schedule believes it is halfway. Windows are memmap slices,
    so winding forward is cheap compared with a step, and it is logged because it is not free.
    """
    if count <= 0:
        return 0
    started = time.perf_counter()
    for _ in range(count):
        next(batches)
    log.info("skipped %d batches in %.1f s to resume", count, time.perf_counter() - started)
    return count


def load_resume(
    resume: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    where: torch.device,
) -> tuple[int, float, str | None]:
    """Restore weights, optimizer and RNG; return ``(start_step, best_val, mlflow run id)``."""
    payload = load_checkpoint(resume, map_location=where)
    start_step = restore(payload, model, optimizer)
    recorded = payload.get("best_val")
    best_val = float(recorded) if isinstance(recorded, int | float) else math.inf
    previous = payload.get("run_id")
    run_id = str(previous) if isinstance(previous, str) and previous else None
    log.info("resumed %s at step %d (mlflow run %s)", resume, start_step, run_id or "new")
    return start_step, best_val, run_id


def write_checkpoint(
    path: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cfg: BaseConfig,
    model_cfg: BaseConfig,
    vocab_hash: str | None,
    manifest_sha: str | None,
    best_val: float,
    run_id: str | None,
) -> Path:
    """One checkpoint with the provenance every ``rukh`` run records."""
    from rukh.tracking import git_sha

    return save_checkpoint(
        path,
        step=step,
        model=model,
        optimizer=optimizer,
        cfg=cfg.model_dump(mode="json"),
        model_cfg=model_cfg.model_dump(mode="json"),
        vocab_hash=vocab_hash,
        data_manifest_sha=manifest_sha,
        git_sha=git_sha(),
        best_val=None if math.isinf(best_val) else best_val,
        run_id=run_id,
    )


def log_metrics(metrics: dict[str, float], step: int) -> None:
    """Send metrics to the active MLflow run; a tracking failure never stops training."""
    import mlflow

    clean = {key: value for key, value in metrics.items() if not math.isnan(value)}
    if not clean:
        return
    try:
        mlflow.log_metrics(clean, step=step)
    except Exception as exc:  # noqa: BLE001 - tracking is not worth a lost run
        log.warning("could not log metrics at step %d: %s", step, exc)
