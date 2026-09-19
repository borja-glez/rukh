"""Staged fine-tuning of the three heads, and the curve that asks how many labels are needed.

Three modes, in the order the lesson walks through them:

``probe``
    the encoder is frozen and only the three linear heads move. It answers the only question
    that matters about the pretraining: *is the information already in the representation?* The
    encoder's weights come out of a probe run bit for bit identical, and a test proves it.
``last-n``
    the last ``last_n`` blocks and the final norm are unfrozen. The early layers keep the general
    features, the late ones specialise.
``full``
    everything moves. Best numbers, most labels needed, easiest to overfit.

On top of that, ``label_curve`` trains the same recipe on 10 %, 25 %, 50 % and 100 % of the
training labels, on **nested** subsets (see ``rukh.data.labels.label_subsets``), so the curve
measures the value of more labels and not the luck of a draw. That curve is the module's lesson:
labelling is the expensive part of this project, and the curve says where it stops paying.

The dataset is the ``squares`` scheme: ``data/evals/positions-eval.parquet`` holds a FEN per
position, not the game that led to it, so a ``moves`` encoder cannot be fine-tuned from it
without joining the games back in — which is a P1 table, and a different job.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from pydantic import Field
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from rukh.config import BaseConfig
from rukh.data.labels import LabelsConfig, build_labels, label_subsets
from rukh.models import EncoderConfig, PositionEncoder
from rukh.models.heads import HEADS, HeadWeights, MultiHead
from rukh.models.squares import fen_to_tokens
from rukh.train.checkpoint import BEST_NAME, load_encoder, step_name
from rukh.train.common import (
    RunConfig,
    autocast_for,
    forever,
    load_resume,
    log_metrics,
    param_groups,
    pick_device,
    run_dir,
    set_lr,
    write_checkpoint,
)

log = logging.getLogger(__name__)

Mode = Literal["probe", "last-n", "full"]
Item = dict[str, Tensor]
CURVE = (0.1, 0.25, 0.5, 1.0)


class LabelledPositions(Dataset[Item]):
    """The labelled rows as tensors; one item is a position and its three labels.

    The FEN is tokenized on demand rather than up front: 69 small integers per row are cheap to
    build and expensive to keep for millions of rows.
    """

    def __init__(self, frame: pl.DataFrame, scheme: Literal["squares", "moves"] = "squares"):
        if scheme != "squares":
            raise ValueError(
                "the supervised table holds a FEN per position, not the game that led to it, "
                "so only the 'squares' scheme can be fine-tuned from it"
            )
        self.fens: list[str] = frame["fen"].to_list()
        self.value = frame["value"].to_numpy().astype("float32")
        self.result = frame["result_class"].to_numpy().astype("int64")
        self.blunder_mask = frame["blunder"].is_not_null().to_numpy()
        self.blunder = frame["blunder"].fill_null(0).to_numpy().astype("float32")

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, index: int) -> Item:
        return {
            "idx": torch.tensor(fen_to_tokens(self.fens[index]), dtype=torch.long),
            "value": torch.tensor(float(self.value[index])),
            "blunder": torch.tensor(float(self.blunder[index])),
            "blunder_mask": torch.tensor(bool(self.blunder_mask[index])),
            "result": torch.tensor(int(self.result[index])),
        }


def make_label_loader(
    dataset: LabelledPositions,
    batch_size: int,
    seed: int = 0,
    workers: int = 0,
    shuffle: bool = True,
) -> DataLoader[Item]:
    """A ``DataLoader`` whose shuffling is fully determined by ``seed``."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        generator=generator,
        persistent_workers=workers > 0,
    )


class HeadsConfig(RunConfig):
    """One fine-tuning run of the three heads; unknown keys in the YAML are an error."""

    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    encoder_ckpt: str | None = None
    """The masked-move checkpoint to start from; ``None`` trains the encoder from scratch,
    which is the honest baseline the pretraining has to beat."""
    model: EncoderConfig | None = None  # only used when there is no checkpoint to load
    input: Literal["squares"] = "squares"
    pooling: Literal["cls", "mean"] = "mean"
    mode: Mode = "probe"
    last_n: int = Field(default=2, ge=1)
    fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    curve: list[float] = list(CURVE)
    weights: HeadWeights = Field(default_factory=HeadWeights)

    @property
    def default_name(self) -> str:
        return f"heads-{self.mode}"

    def encoder(self) -> EncoderConfig:
        """The encoder config used when no checkpoint is loaded."""
        base = self.model if self.model is not None else EncoderConfig()
        return base.model_copy(update={"input": self.input})


class HeadsResult(BaseConfig):
    """What one fine-tuning run produced: where the weights are and how they scored."""

    checkpoint: str
    mode: Mode
    fraction: float
    train_labels: int
    val_labels: int
    metrics: dict[str, float]


def freeze_encoder(model: MultiHead, mode: Mode, last_n: int = 2) -> int:
    """Apply a fine-tuning mode and return how many encoder tensors stay trainable."""
    for param in model.encoder.parameters():
        param.requires_grad = mode == "full"
    if mode == "last-n":
        blocks = list(model.encoder.blocks)
        for block in blocks[max(0, len(blocks) - last_n) :]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.encoder.ln_f.parameters():
            param.requires_grad = True
    return sum(param.requires_grad for param in model.encoder.parameters())


def set_training_mode(model: MultiHead, mode: Mode) -> None:
    """Training mode for the whole model, with the frozen encoder kept in ``eval``.

    A frozen encoder in ``train`` mode would still apply dropout, so the same position would
    give the heads a different vector every epoch: noise the heads cannot learn away and the
    encoder cannot absorb, because it is not learning at all.
    """
    model.train()
    if mode == "probe":
        model.encoder.eval()


def build_frames(cfg: HeadsConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(train, val)`` label frames, split by ``game_id`` and cut to ``cfg.fraction``."""
    frame = build_labels(cfg.labels)
    train = frame.filter(pl.col("split") == "train")
    if cfg.fraction < 1.0:
        train = label_subsets(train, [cfg.fraction])[cfg.fraction]
    return train, frame.filter(pl.col("split") == "val")


def load_encoder_for(cfg: HeadsConfig, where: torch.device) -> PositionEncoder:
    """The pretrained encoder of ``encoder_ckpt``, or a fresh one when there is none."""
    if cfg.encoder_ckpt is None:
        return PositionEncoder(cfg.encoder())
    from rukh import paths

    encoder, _payload = load_encoder(paths.resolve(cfg.encoder_ckpt), map_location=where)
    if encoder.cfg.input != cfg.input:
        raise ValueError(
            f"{cfg.encoder_ckpt} was trained on the {encoder.cfg.input!r} scheme, "
            f"the run asks for {cfg.input!r}"
        )
    return encoder


def to_device(batch: Item, where: torch.device) -> Item:
    return {key: value.to(where) for key, value in batch.items()}


@torch.no_grad()
def evaluate_heads(
    model: MultiHead, loader: DataLoader[Item], batches: int, where: torch.device
) -> dict[str, float]:
    """Loss and one metric per head over at most ``batches`` validation batches."""
    was_training = model.training
    model.eval()
    totals = {f"val/{name}_loss": 0.0 for name in HEADS}
    seen = 0
    value_error = 0.0
    blunder_hits = blunder_seen = 0
    result_hits = 0
    for index, raw in enumerate(loader):
        if index >= batches:
            break
        batch = to_device(raw, where)
        outputs = model(batch["idx"])
        total, parts = model.loss(outputs, batch)
        rows = int(batch["idx"].shape[0])
        seen += rows
        totals["val/loss"] = totals.get("val/loss", 0.0) + float(total) * rows
        for name in HEADS:
            totals[f"val/{name}_loss"] += float(parts[name]) * rows
        value_error += float((outputs["value"] - batch["value"]).abs().sum())
        mask = batch["blunder_mask"].bool()
        if bool(mask.any()):
            predicted = (outputs["blunder"][mask] > 0).float()
            blunder_hits += int((predicted == batch["blunder"][mask]).sum())
            blunder_seen += int(mask.sum())
        result_hits += int((outputs["result"].argmax(dim=-1) == batch["result"]).sum())
    if was_training:
        model.train()
    if not seen:
        return {"val/loss": math.nan}
    metrics = {key: value / seen for key, value in totals.items()}
    metrics["val/value_mae"] = value_error / seen
    metrics["val/blunder_acc"] = blunder_hits / blunder_seen if blunder_seen else math.nan
    metrics["val/result_acc"] = result_hits / seen
    return metrics


def train_heads(
    cfg: HeadsConfig, resume: Path | None = None, device: str | None = None
) -> HeadsResult:
    """Fine-tune the three heads over a (possibly frozen) encoder and return the run's result."""
    cfg.check()
    torch.manual_seed(cfg.seed)
    where = torch.device(device or pick_device())
    train_frame, val_frame = build_frames(cfg)
    if not train_frame.height:
        raise ValueError("the training split is empty; check labels.val_fraction and the source")
    train_loader = make_label_loader(
        LabelledPositions(train_frame, cfg.input), cfg.batch_size, cfg.seed, cfg.workers
    )
    val_loader = make_label_loader(
        LabelledPositions(val_frame, cfg.input), cfg.batch_size, cfg.seed, 0, shuffle=False
    )

    encoder = load_encoder_for(cfg, where)
    model = MultiHead(encoder, cfg.weights, cfg.pooling).to(where)
    trainable = freeze_encoder(model, cfg.mode, cfg.last_n)
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    start_step = 0
    best_val = math.inf
    run_id: str | None = None
    if resume is not None:
        start_step, best_val, run_id = load_resume(resume, model, optimizer, where)
    autocast, _ = autocast_for(cfg, where)
    set_training_mode(model, cfg.mode)

    out_dir = run_dir(cfg, resume)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = forever(train_loader)
    final = out_dir / step_name(cfg.max_steps)
    metrics: dict[str, float] = {}

    from rukh.tracking import start_run

    params = {
        **cfg.model_dump(mode="json"),
        "device": str(where),
        "train_labels": train_frame.height,
        "val_labels": val_frame.height,
        "trainable_encoder_tensors": trainable,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    tags = {"objective": "heads", "mode": cfg.mode, "fraction": str(cfg.fraction)}
    with start_run(out_dir.name, params, tags=tags, run_id=run_id) as run:
        log.info("run %s in %s on %s (%s)", run.info.run_id, out_dir, where, cfg.mode)
        this_run = str(run.info.run_id)

        def save(path: Path, step: int) -> Path:
            return write_checkpoint(
                path,
                step=step,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                model_cfg=encoder.cfg,
                vocab_hash=None,
                manifest_sha=None,
                best_val=best_val,
                run_id=this_run,
            )

        clock = time.perf_counter()
        for step in range(start_step, cfg.max_steps):
            lr = set_lr(optimizer, step, cfg)
            optimizer.zero_grad(set_to_none=True)
            step_losses = {name: 0.0 for name in HEADS}
            step_total = 0.0
            for _ in range(cfg.grad_accum):
                batch = to_device(next(batches), where)
                with autocast:
                    outputs = model(batch["idx"])
                    total, parts = model.loss(outputs, batch)
                (total / cfg.grad_accum).backward()
                step_total += float(total.detach()) / cfg.grad_accum
                for name in HEADS:
                    step_losses[name] += float(parts[name].detach()) / cfg.grad_accum
            grad_norm = float(
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
                )
            )
            optimizer.step()

            done = step + 1
            last = done == cfg.max_steps
            if done % cfg.log_every == 0 or last:
                elapsed = max(time.perf_counter() - clock, 1e-9)
                steps = min(cfg.log_every, done - start_step)
                log_metrics(
                    {
                        "train/loss": step_total,
                        **{f"train/{name}_loss": step_losses[name] for name in HEADS},
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "positions_per_s": cfg.batch_size * cfg.grad_accum * steps / elapsed,
                    },
                    step=done,
                )
                clock = time.perf_counter()
            if done % cfg.eval_every == 0 or last:
                metrics = evaluate_heads(model, val_loader, cfg.eval_batches, where)
                set_training_mode(model, cfg.mode)
                log_metrics(metrics, step=done)
                val_loss = metrics.get("val/loss", math.nan)
                log.info("step %d  val/loss %.4f", done, val_loss)
                if not math.isnan(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    save(out_dir / BEST_NAME, done)
                clock = time.perf_counter()
            if done % cfg.ckpt_every == 0 or last:
                final = save(out_dir / step_name(done), done)
                clock = time.perf_counter()
    return HeadsResult(
        checkpoint=str(final),
        mode=cfg.mode,
        fraction=cfg.fraction,
        train_labels=train_frame.height,
        val_labels=val_frame.height,
        metrics=metrics,
    )


def label_curve(
    cfg: HeadsConfig, fractions: list[float] | None = None, device: str | None = None
) -> list[HeadsResult]:
    """Train the same recipe on nested subsets of the labels; one result per fraction."""
    wanted = sorted(set(fractions if fractions is not None else cfg.curve))
    results: list[HeadsResult] = []
    for fraction in wanted:
        run = cfg.model_copy(
            update={
                "fraction": fraction,
                "run_name": f"{cfg.run_name or cfg.default_name}-{int(fraction * 100)}",
            }
        )
        results.append(train_heads(run, device=device))
    return results
