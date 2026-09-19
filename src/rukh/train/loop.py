"""The decoder's training loop: next-token prediction on a packed stream of move tokens.

The recipe follows ``docs/spec/02`` (component 1): AdamW (0.9/0.95, weight decay 0.1 applied
only to matrices), learning rate 6e-4 with 1 000 warmup steps and a cosine decay, an effective
batch of ``batch_size * grad_accum`` sequences, bf16 autocast on CUDA, gradient clipping at 1.0
and optional ``torch.compile``. Everything the run needs to be reproducible (config, seed, git
SHA, vocabulary hash, data manifest hash) goes to MLflow and into every checkpoint. All of that
machinery lives in ``rukh.train.common``, shared with the encoder's masked-move loop; what is
left here is the part that is specific to predicting the next move.

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
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from rukh import paths
from rukh.models import DecoderConfig, MoveDecoder, preset
from rukh.tokenize.loader import IGNORE_INDEX, PackedDataset, make_loader
from rukh.train.checkpoint import BEST_NAME, read_manifest_sha, read_vocab_hash, step_name
from rukh.train.common import (
    Batch,
    RunConfig,
    autocast_for,
    forever,
    load_resume,
    log_metrics,
    maybe_compile,
    param_groups,
    pick_device,
    run_dir,
    set_lr,
    skip_batches,
    write_checkpoint,
)

log = logging.getLogger(__name__)

__all__ = [
    "Batch",
    "TrainConfig",
    "evaluate",
    "forever",
    "log_metrics",
    "maybe_compile",
    "param_groups",
    "pick_device",
    "run_dir",
    "skip_batches",
    "train",
]


class TrainConfig(RunConfig):
    """Everything one decoder training run needs; unknown keys in the YAML are an error."""

    preset: Literal["tiny", "small", "medium"] = "small"
    model: DecoderConfig | None = None  # overrides the preset when given

    @property
    def default_name(self) -> str:
        return self.preset

    def decoder(self) -> DecoderConfig:
        """The decoder config of this run: the preset (or ``model``) with ``block`` applied."""
        base = self.model if self.model is not None else preset(self.preset)
        return base.model_copy(update={"block": self.block})


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


def train(cfg: TrainConfig, resume: Path | None = None, device: str | None = None) -> Path:
    """Train a ``MoveDecoder`` and return the path of the final checkpoint."""
    cfg.check()
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
        start_step, best_val, run_id = load_resume(resume, model, optimizer, where)

    autocast, use_bf16 = autocast_for(cfg, where)
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

    from rukh.tracking import start_run

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
            return write_checkpoint(
                path,
                step=step,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                model_cfg=model_cfg,
                vocab_hash=vocab_hash,
                manifest_sha=manifest_sha,
                best_val=best_val,
                run_id=this_run,
            )

        clock = time.perf_counter()
        for step in range(start_step, cfg.max_steps):
            lr = set_lr(optimizer, step, cfg)
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
