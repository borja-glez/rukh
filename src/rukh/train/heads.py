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

Both input schemes can be fine-tuned here, which is what makes the spec's comparison possible:

``squares``
    ``data/evals/positions-eval.parquet`` holds a FEN per position, and 69 fixed tokens come
    straight out of it.
``moves``
    the position is the *line that reached it*: ``rukh.data.labels.game_moves`` joins the P1
    games back in and the item is ``[<bos>, <wXXXX>, <bXXXX>] + uci.split()[:ply]``, cropped
    header-first by ``rukh.infer.sampler.prompt_ids`` and padded per batch. This is the scheme
    the masked-move pretraining runs on, so it is the only one an MMM checkpoint can be loaded
    into — before, the pretrained encoder had nowhere to go.

    The caveat of that path: the supervised table is deduplicated by ``fen4``, so the prefix is
    **a** line reaching the position, not necessarily the labelled game's. The position, the
    value and the blunder verdict are the same either way; the history may not be.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from pydantic import Field
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from rukh.config import BaseConfig
from rukh.data.labels import LabelsConfig, build_labels, game_moves, label_subsets
from rukh.infer.sampler import HEADER_TOKENS, prompt_ids
from rukh.models import EncoderConfig, PositionEncoder
from rukh.models.encoder import PAD_ID
from rukh.models.heads import HEADS, HeadWeights, MultiHead
from rukh.models.squares import fen_to_tokens
from rukh.tokenize.uci_vocab import UciTokenizer, elo_token
from rukh.train.checkpoint import BEST_NAME, CURVE_KEY, attach_payload, load_encoder, step_name
from rukh.train.common import (
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
    write_checkpoint,
)

log = logging.getLogger(__name__)

Mode = Literal["probe", "last-n", "full"]
Scheme = Literal["squares", "moves"]
Item = dict[str, Tensor]
CURVE = (0.1, 0.25, 0.5, 1.0)


class LabelledPositions(Dataset[Item]):
    """The labelled rows as tensors; one item is a position and its three labels.

    The tokens are built on demand rather than up front: a handful of small integers per row is
    cheap to build and expensive to keep for millions of rows. On ``squares`` that is the 69
    tokens of the FEN; on ``moves`` it is the header plus the first ``ply`` moves of the game,
    which is why ``moves`` needs the ``(game_id, uci)`` frame of
    ``rukh.data.labels.game_moves`` — stored per **game**, never copied per row.
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        scheme: Scheme = "squares",
        moves: pl.DataFrame | None = None,
        block: int = 200,
    ):
        self.scheme = scheme
        self.block = block
        self.value = frame["value"].to_numpy().astype("float32")
        self.result = frame["result_class"].to_numpy().astype("int64")
        self.blunder_mask = frame["blunder"].is_not_null().to_numpy()
        self.blunder = frame["blunder"].fill_null(0).to_numpy().astype("float32")
        self.fens: list[str] = frame["fen"].to_list()
        self.game_ids: list[int] = [int(value) for value in frame["game_id"].to_list()]
        self.plies: list[int] = [int(value) for value in frame["ply"].to_list()]
        self.prefixes: dict[int, list[int]] = {}
        if scheme == "moves":
            if moves is None:
                raise ValueError(
                    "the 'moves' scheme needs the games of the labelled rows: pass the frame "
                    "of rukh.data.labels.game_moves(frame, cfg.labels.games_dir)"
                )
            self.prefixes = _move_prefixes(moves)
            unknown = {game for game in self.game_ids if game not in self.prefixes}
            if unknown:
                raise ValueError(
                    f"{len(unknown)} labelled games are missing from the games frame "
                    f"(first: {sorted(unknown)[:3]}); check labels.games_dir"
                )

    def __len__(self) -> int:
        return len(self.fens)

    def tokens(self, index: int) -> list[int]:
        """The input ids of one row, in the scheme this dataset was built for."""
        if self.scheme == "squares":
            return fen_to_tokens(self.fens[index])
        prefix = self.prefixes[self.game_ids[index]]
        # ``ply`` counts the move that *led to* this position, so the prefix includes it.
        ids = prefix[: HEADER_TOKENS + self.plies[index]]
        # A prefix longer than the context keeps <bos> and the two Elo tokens and drops the
        # oldest moves: the same header-first crop the sampler uses, so the fine-tuned model
        # sees the shape the pretraining and the demo do.
        return prompt_ids(ids, self.block)

    def __getitem__(self, index: int) -> Item:
        return {
            "idx": torch.tensor(self.tokens(index), dtype=torch.long),
            "value": torch.tensor(float(self.value[index])),
            "blunder": torch.tensor(float(self.blunder[index])),
            "blunder_mask": torch.tensor(bool(self.blunder_mask[index])),
            "result": torch.tensor(int(self.result[index])),
        }


def _move_prefixes(moves: pl.DataFrame) -> dict[int, list[int]]:
    """``game_id -> [<bos>, elo(white), elo(black), *every move]``, one entry per game."""
    tok = UciTokenizer()
    out: dict[int, list[int]] = {}
    for game_id, uci, white_elo, black_elo in moves.select(
        "game_id", "uci", "white_elo", "black_elo"
    ).rows():
        ids = [
            tok.bos_id,
            tok.vocab[elo_token(int(white_elo), "w")],
            tok.vocab[elo_token(int(black_elo), "b")],
        ]
        ids.extend(tok.vocab.get(move, tok.unk_id) for move in str(uci).split())
        out[int(game_id)] = ids
    return out


def collate(items: list[Item]) -> Item:
    """Pad a batch to its longest sequence and say which tokens are real.

    ``squares`` items are all 69 tokens long and the mask is all ``True``; ``moves`` items are
    as long as the game was, so the short ones are padded with ``<pad>`` and the mask keeps the
    encoder's attention off them. The padding is never silent: ``attention_mask`` travels with
    the batch and ``PositionEncoder.pool`` averages the real tokens only.
    """
    width = max(int(item["idx"].shape[0]) for item in items)
    idx = torch.full((len(items), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.bool)
    for row, item in enumerate(items):
        tokens = item["idx"]
        idx[row, : tokens.shape[0]] = tokens
        mask[row, : tokens.shape[0]] = True
    batch: Item = {"idx": idx, "attention_mask": mask}
    for key in ("value", "blunder", "blunder_mask", "result"):
        # The labels are optional so that inference (``rukh eval encoder``) can pad a batch of
        # bare token sequences with this very function instead of a second copy of it.
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items])
    return batch


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
        collate_fn=collate,
    )


class HeadsConfig(RunConfig):
    """One fine-tuning run of the three heads; unknown keys in the YAML are an error."""

    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    encoder_ckpt: str | None = None
    """The masked-move checkpoint to start from; ``None`` trains the encoder from scratch,
    which is the honest baseline the pretraining has to beat."""
    model: EncoderConfig | None = None  # only used when there is no checkpoint to load
    input: Scheme = "squares"
    """The representation the heads are fine-tuned on; it must match ``encoder_ckpt``'s."""
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
        return base.model_copy(update={"input": self.input, "block": self.block})


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


def set_training_mode(model: MultiHead, mode: Mode, last_n: int = 2) -> None:
    """Training mode for the whole model, with every **frozen** block kept in ``eval``.

    A frozen block in ``train`` mode would still apply dropout, so the same position would give
    the heads a different vector every epoch: noise the heads cannot learn away and the block
    cannot absorb, because it is not learning at all. Under ``last-n`` that applies to the
    frozen prefix too — the early blocks and the embeddings go to ``eval`` and only the last
    ``last_n`` blocks and the final norm keep their dropout, which is the regularisation of the
    part that is actually being trained.
    """
    model.train()
    if mode == "probe":
        model.encoder.eval()
    elif mode == "last-n":
        encoder = model.encoder
        encoder.eval()
        for block in list(encoder.blocks)[max(0, len(encoder.blocks) - last_n) :]:
            block.train()
        encoder.ln_f.train()


def build_frames(cfg: HeadsConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(train, val)`` label frames, split by ``game_id`` and cut to ``cfg.fraction``."""
    frame = build_labels(cfg.labels)
    train = frame.filter(pl.col("split") == "train")
    if cfg.fraction < 1.0:
        train = label_subsets(train, [cfg.fraction])[cfg.fraction]
    return train, frame.filter(pl.col("split") == "val")


def build_datasets(
    cfg: HeadsConfig, train_frame: pl.DataFrame, val_frame: pl.DataFrame
) -> tuple[LabelledPositions, LabelledPositions]:
    """The two datasets of a run; on ``moves`` the games are read once and shared."""
    moves = None
    if cfg.input == "moves":
        moves = game_moves(pl.concat([train_frame, val_frame]), cfg.labels.games_dir)
        log.info("%d games joined back for the 'moves' scheme", moves.height)
    return (
        LabelledPositions(train_frame, cfg.input, moves, cfg.block),
        LabelledPositions(val_frame, cfg.input, moves, cfg.block),
    )


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


def _compile_probe(
    model: MultiHead, dataset: LabelledPositions, where: torch.device, batch_size: int
) -> Callable[[nn.Module], None]:
    """One real step, so ``torch.compile`` runs here and not in the middle of the loop.

    The probe uses the dataset's own items, mask included: probing with ``attention_mask=None``
    while the loop always passes one would compile a graph the loop never runs and pay for a
    recompilation at step 1.
    """

    def probe(compiled: nn.Module) -> None:
        rows = [dataset[index] for index in range(min(batch_size, len(dataset)))]
        batch = to_device(collate(rows), where)
        outputs = compiled(batch["idx"], batch["attention_mask"])
        total, _ = model.loss(outputs, batch)
        total.backward()

    return probe


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
        outputs = model(batch["idx"], batch.get("attention_mask"))
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
    train_set, val_set = build_datasets(cfg, train_frame, val_frame)
    train_loader = make_label_loader(train_set, cfg.batch_size, cfg.seed, cfg.workers)
    val_loader = make_label_loader(val_set, cfg.batch_size, cfg.seed, 0, shuffle=False)

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
    with autocast:  # compile under the same precision the loop will use
        probe = _compile_probe(model, train_set, where, cfg.batch_size)
        runnable = maybe_compile(model, cfg.compile, probe=probe)
    set_training_mode(model, cfg.mode, cfg.last_n)

    out_dir = run_dir(cfg, resume)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = forever(train_loader)
    final = out_dir / step_name(cfg.max_steps)
    metrics: dict[str, float] = {}

    from rukh.tracking import start_run

    params = {
        # ``tokens_dir`` is the one inherited knob this loop has no use for: the labels are a
        # parquet of positions, not a packed token stream.
        **cfg.model_dump(mode="json", exclude={"tokens_dir"}),
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
                    outputs = runnable(batch["idx"], batch.get("attention_mask"))
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
                set_training_mode(model, cfg.mode, cfg.last_n)
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
    """Train the same recipe on nested subsets of the labels; one result per fraction.

    The points are written back into the checkpoints of the **last** run (the one with the most
    labels, which is the checkpoint anybody would then evaluate or publish), so
    ``rukh eval encoder`` can render the "labels needed" table without being told where the
    other runs are. Without that write the curve would exist only in MLflow and in this return
    value, and the table — a ``docs/acceptance.md`` deliverable — could never be filled.
    """
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
    if results:
        write_curve(Path(results[-1].checkpoint).parent, results)
    return results


def write_curve(run_dir: Path, results: Sequence[HeadsResult]) -> list[Path]:
    """Record the curve in every checkpoint of ``run_dir``; returns the files it touched."""
    points = [result.model_dump(mode="json") for result in results]
    written: list[Path] = []
    for path in sorted(run_dir.glob("*.pt")):
        attach_payload(path, **{CURVE_KEY: points})
        written.append(path)
    log.info("label curve of %d points written into %d checkpoints", len(points), len(written))
    return written
