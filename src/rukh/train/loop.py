"""The training loop: AdamW with a cosine schedule, bf16, checkpoints and MLflow logging.

The recipe follows ``docs/spec/02`` (component 1): AdamW (0.9/0.95, weight decay 0.1 applied
only to matrices), learning rate 6e-4 with 1 000 warmup steps and a cosine decay, an effective
batch of ``batch_size * grad_accum`` sequences, bf16 autocast on CUDA, gradient clipping at 1.0
and optional ``torch.compile``. Everything the run needs to be reproducible (config, seed, git
SHA, vocabulary hash, data manifest hash) goes to MLflow and into every checkpoint.

Resuming is meant to be indistinguishable from never having stopped: the batch stream is wound
forward past the windows the first half of the run already saw, and the MLflow run id travels in
the checkpoint so the curve carries on in the same run instead of starting a second one.

Two throughput numbers are logged because they answer different questions: ``tokens_per_s``
counts every position in the window (what the GPU actually processed, comparable across runs)
and ``real_tokens_per_s`` counts only the non-``<pad>`` targets (what the model learned from).
The training loss is accumulated token-weighted rather than as a mean of means, so a
micro-batch with fewer real tokens does not count as much as a full one.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Iterator
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from rukh import paths
from rukh.config import BaseConfig
from rukh.models import DecoderConfig, MoveDecoder, preset
from rukh.tokenize.loader import IGNORE_INDEX, PackedDataset, make_loader
from rukh.train.checkpoint import (
    BEST_NAME,
    load_checkpoint,
    read_manifest_sha,
    read_vocab_hash,
    restore,
    save_checkpoint,
    step_name,
)
from rukh.train.schedule import lr_at

log = logging.getLogger(__name__)

Batch = tuple[torch.Tensor, torch.Tensor]


class TrainConfig(BaseConfig):
    """Everything one training run needs; unknown keys in the YAML are an error."""

    preset: Literal["tiny", "small", "medium"] = "small"
    model: DecoderConfig | None = None  # overrides the preset when given
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

    def decoder(self) -> DecoderConfig:
        """The decoder config of this run: the preset (or ``model``) with ``block`` applied."""
        base = self.model if self.model is not None else preset(self.preset)
        return base.model_copy(update={"block": self.block})


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


def forever(loader: DataLoader[Batch]) -> Iterator[Batch]:
    """Repeat a loader for as many steps as the schedule asks for."""
    while True:
        yield from loader


def maybe_compile(
    model: MoveDecoder, enabled: bool, sample: torch.Tensor | None = None
) -> nn.Module:
    """``torch.compile`` the model when asked; a failure is a warning, never a stopped run.

    ``torch.compile`` is lazy: on Windows without MSVC it only fails when the first forward
    reaches Inductor. ``sample`` (one batch of the training shape) forces that compilation here,
    where it can still fall back to eager, and Dynamo's own error suppression covers any later
    recompilation for a different shape.
    """
    if not enabled:
        return model
    try:
        import torch._dynamo as dynamo

        dynamo.config.suppress_errors = True
        compiled = torch.compile(model)
        if sample is not None:
            _, loss = compiled(sample, sample)
            if loss is not None:
                loss.backward()
            model.zero_grad(set_to_none=True)
        return compiled
    except Exception as exc:  # noqa: BLE001 - compilation backends fail in many ways
        log.warning("torch.compile is unavailable, training eagerly: %s", exc)
        model.zero_grad(set_to_none=True)
        return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Batch],
    batches: int,
    device: torch.device,
    autocast: Any = None,
) -> tuple[float, float]:
    """Validation loss and top-1 next-token accuracy over at most ``batches`` batches.

    The loss is token-weighted: each batch's mean is weighted by the number of non-``<pad>``
    targets it had, so a short last batch does not count as much as a full one.
    """
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    weighted = 0
    hits = 0
    counted = 0
    for index, (x, y) in enumerate(loader):
        if index >= batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast if autocast is not None else nullcontext():
            logits, loss = model(x, y)
        mask = y != IGNORE_INDEX
        tokens = int(mask.sum())
        if loss is not None and torch.isfinite(loss) and tokens:
            loss_sum += loss.float().item() * tokens
            weighted += tokens
        hits += int((logits.argmax(dim=-1) == y)[mask].sum())
        counted += tokens
    if was_training:
        model.train()
    return (loss_sum / weighted if weighted else math.nan, hits / counted if counted else 0.0)


def run_dir(cfg: TrainConfig, resume: Path | None = None) -> Path:
    """Where this run writes: the resumed run's folder, or ``out_dir/<run name>``.

    The name carries a timestamp unless ``unique_run_name`` is off, because two runs of the same
    config would otherwise write the same ``step-*.pt`` files and the second would quietly
    overwrite the checkpoint series of the first.
    """
    if resume is not None:
        return Path(resume).resolve().parent
    name = cfg.run_name or cfg.preset
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


def train(cfg: TrainConfig, resume: Path | None = None, device: str | None = None) -> Path:
    """Train a ``MoveDecoder`` and return the path of the final checkpoint."""
    if min(cfg.max_steps, cfg.grad_accum, cfg.batch_size) < 1:
        raise ValueError("max_steps, grad_accum and batch_size must be positive")
    torch.manual_seed(cfg.seed)
    where = torch.device(device or pick_device())
    tokens_dir = paths.resolve(cfg.tokens_dir)
    model_cfg = cfg.decoder()

    train_set = PackedDataset(tokens_dir / "train", block=cfg.block)
    val_set = PackedDataset(tokens_dir / "val", block=cfg.block)
    if train_set.info.vocab_size != model_cfg.vocab_size:
        raise ValueError(
            f"{tokens_dir / 'train'} has vocab_size {train_set.info.vocab_size}, "
            f"the model expects {model_cfg.vocab_size}"
        )
    train_loader = make_loader(train_set, cfg.batch_size, seed=cfg.seed, workers=cfg.workers)
    val_loader = make_loader(
        val_set, cfg.batch_size, seed=cfg.seed, workers=0, shuffle=False, drop_last=False
    )
    if not len(train_loader):
        raise ValueError(f"{tokens_dir / 'train'} has fewer than {cfg.batch_size} windows")

    model = MoveDecoder(model_cfg).to(where)
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    start_step = 0
    best_val = math.inf
    run_id: str | None = None
    if resume is not None:
        payload = load_checkpoint(resume, map_location=where)
        start_step = restore(payload, model, optimizer)
        recorded = payload.get("best_val")
        best_val = float(recorded) if isinstance(recorded, int | float) else best_val
        previous = payload.get("run_id")
        run_id = str(previous) if isinstance(previous, str) and previous else None
        log.info("resumed %s at step %d (mlflow run %s)", resume, start_step, run_id or "new")

    use_bf16 = cfg.precision == "bf16" and where.type == "cuda"
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    )
    warmup = torch.ones((cfg.batch_size, cfg.block), dtype=torch.long, device=where)
    with autocast:  # compile under the same precision the loop will use
        runnable = maybe_compile(model, cfg.compile, warmup)
    runnable.train()

    out_dir = run_dir(cfg, resume)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_hash = read_vocab_hash(tokens_dir / "train")
    manifest_sha = read_manifest_sha(paths.data_dir() / "raw" / "manifest.json")
    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.block
    batches = forever(train_loader)
    skip_batches(batches, start_step * cfg.grad_accum)
    final = out_dir / step_name(cfg.max_steps)

    from rukh.tracking import git_sha, start_run

    params = {
        **cfg.model_dump(mode="json"),
        "vocab_hash": vocab_hash,
        "data_manifest_sha": manifest_sha,
        "device": str(where),
        "num_params": model.num_params(),
    }
    with start_run(out_dir.name, params, tags={"preset": cfg.preset}, run_id=run_id) as run:
        log.info("run %s in %s on %s", run.info.run_id, out_dir, where)
        this_run = str(run.info.run_id)

        def save(path: Path, step: int) -> Path:
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
                run_id=this_run,
            )

        clock = time.perf_counter()
        for step in range(start_step, cfg.max_steps):
            lr = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            # Token-weighted, on the device: one synchronisation per step instead of one per
            # micro-batch, and a micro-batch with fewer real tokens weighs less in the mean.
            loss_sum = torch.zeros((), device=where)
            real_tokens = torch.zeros((), device=where)
            for _ in range(cfg.grad_accum):
                x, y = next(batches)
                x, y = x.to(where), y.to(where)
                with autocast:
                    _, loss = runnable(x, y)
                assert loss is not None
                (loss / cfg.grad_accum).backward()
                tokens = (y != IGNORE_INDEX).sum()
                loss_sum += loss.detach().float() * tokens
                real_tokens += tokens
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            optimizer.step()

            done = step + 1
            last = done == cfg.max_steps
            if done % cfg.log_every == 0 or last:
                counted = float(real_tokens.item())
                total = float(loss_sum.item()) / counted if counted else math.nan
                elapsed = max(time.perf_counter() - clock, 1e-9)
                steps = min(cfg.log_every, done - start_step)
                log_metrics(
                    {
                        "train/loss": total,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "tokens_per_s": tokens_per_step * steps / elapsed,
                        "real_tokens_per_s": counted * steps / elapsed,
                    },
                    step=done,
                )
                clock = time.perf_counter()
            if done % cfg.eval_every == 0 or last:
                val_loss, top1 = evaluate(
                    runnable, val_loader, cfg.eval_batches, where, autocast if use_bf16 else None
                )
                log_metrics({"val/loss": val_loss, "val/top1": top1}, step=done)
                log.info("step %d  val/loss %.4f  val/top1 %.4f", done, val_loss, top1)
                if not math.isnan(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    save(out_dir / BEST_NAME, done)
                clock = time.perf_counter()
            if done % cfg.ckpt_every == 0 or last:
                final = save(out_dir / step_name(done), done)
                clock = time.perf_counter()
    return final


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
