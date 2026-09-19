"""Masked move modeling: BERT's pretraining objective moved from words to chess moves.

A fraction of the moves of a game are hidden and the bidirectional encoder has to put them back
from both sides of the hole. The 80/10/10 recipe is BERT's and it is kept for the same reason:
if every hidden position were replaced by ``<mask>``, the model would only ever see ``<mask>``
during pretraining and never during fine-tuning, and it would learn a representation that only
works when a token is missing. Ten per cent of random tokens force it to distrust what it reads
(a move that is *there* may still be wrong) and ten per cent left untouched force it to build a
representation of every position, not only of the marked ones.

Two things differ from the decoder's loop, and only two:

* control tokens are never hidden. ``<bos>``, the two Elo tokens, the result token and
  ``<eos>`` are the *condition* of the game, not the signal; predicting the result from a full
  game would be free and would teach nothing about chess.
* the ignored label is ``-100``, not the decoder's ``0``. Here ``0`` is ``<pad>``, a token the
  ``squares`` scheme can legitimately be asked to predict, so "nothing to predict" needs a value
  outside every vocabulary. See ``rukh.models.encoder.MMM_IGNORE_INDEX``.

Everything else — optimizer, schedule, checkpoints, resume, MLflow — is ``rukh.train.common``,
the same machinery the decoder uses, so the two runs are comparable.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rukh import paths
from rukh.config import BaseConfig
from rukh.models import EncoderConfig, PositionEncoder
from rukh.models.encoder import MMM_IGNORE_INDEX
from rukh.models.squares import CONTROL_IDS as SQUARE_CONTROL_IDS
from rukh.models.squares import MASK_ID as SQUARE_MASK_ID
from rukh.tokenize.loader import PackedDataset, make_loader
from rukh.tokenize.uci_vocab import MASK_ID as MOVE_MASK_ID
from rukh.tokenize.uci_vocab import SPECIALS, elo_tokens
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

Scheme = Literal["moves", "squares"]

MOVE_CONTROL_IDS: frozenset[int] = frozenset(
    range(len(SPECIALS) + len(elo_tokens("w")) + len(elo_tokens("b")))
)
"""The 62 ids of the UCI vocabulary that are not moves: specials, results and the Elo bins.

They are a prefix of the vocabulary by construction (see ``uci_vocab.build_vocab``), which is
what lets a random replacement simply draw above them.
"""


def control_ids(scheme: Scheme) -> frozenset[int]:
    """Token ids that are never hidden, for one input scheme."""
    return MOVE_CONTROL_IDS if scheme == "moves" else SQUARE_CONTROL_IDS


def mask_id(scheme: Scheme) -> int:
    """The ``<mask>`` token of one input scheme (id 3 in the P1 vocabulary, 1 in squares)."""
    return MOVE_MASK_ID if scheme == "moves" else SQUARE_MASK_ID


class MaskingConfig(BaseConfig):
    """How much to hide and how: BERT's 15 % at 80/10/10."""

    prob: float = 0.15
    mask_ratio: float = 0.8
    random_ratio: float = 0.1
    keep_ratio: float = 0.1
    seed: int = 0

    @model_validator(mode="after")
    def _check(self) -> MaskingConfig:
        if not 0.0 < self.prob <= 1.0:
            raise ValueError(f"prob must be in (0, 1], got {self.prob}")
        ratios = (self.mask_ratio, self.random_ratio, self.keep_ratio)
        if any(ratio < 0.0 for ratio in ratios):
            raise ValueError("mask_ratio, random_ratio and keep_ratio must be non-negative")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError(f"mask/random/keep must add up to 1, got {sum(ratios)}")
        return self


def masking_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """A generator for ``apply_masking``: one per run, so the masking is seeded but varies."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def apply_masking(
    batch: Tensor,
    cfg: MaskingConfig,
    vocab: int,
    scheme: Scheme = "moves",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Hide part of ``batch``; return ``(inputs, labels)`` of the same shape.

    ``labels`` is the original token where a position was selected and ``MMM_IGNORE_INDEX``
    everywhere else, so the loss scores exactly the selected positions and nothing else. Of the
    selected ones, ``mask_ratio`` become ``<mask>``, ``random_ratio`` become another *move*
    (never a control token) and ``keep_ratio`` are left as they were.

    ``generator`` makes the draw reproducible; without one the global torch RNG is used, which
    the training loop seeds once from ``RunConfig.seed``.
    """
    control = control_ids(scheme)
    if vocab <= max(control):
        raise ValueError(f"a vocabulary of {vocab} tokens is all control tokens")
    device = batch.device
    is_control = torch.isin(batch, torch.tensor(sorted(control), device=device))
    draw = torch.rand(batch.shape, generator=generator, device=device)
    selected = (draw < cfg.prob) & ~is_control
    labels = torch.where(selected, batch, torch.full_like(batch, MMM_IGNORE_INDEX))

    decision = torch.rand(batch.shape, generator=generator, device=device)
    inputs = batch.clone()
    inputs[selected & (decision < cfg.mask_ratio)] = mask_id(scheme)
    random_upto = cfg.mask_ratio + cfg.random_ratio
    replace = selected & (decision >= cfg.mask_ratio) & (decision < random_upto)
    if bool(replace.any()):
        # Draw above the control block: a random *move*, never a fake Elo or result token.
        low = max(control) + 1
        noise = torch.randint(
            low, vocab, (int(replace.sum()),), generator=generator, device=device, dtype=batch.dtype
        )
        inputs[replace] = noise
    return inputs, labels


class MmmConfig(RunConfig):
    """One masked-move pretraining run; unknown keys in the YAML are an error.

    ``tokens_dir`` is a packed *move* stream, the very one the decoder trains on, so the two
    models see the same games. The ``squares`` scheme has no pack in P1 (positions live in a
    parquet of FENs, not in a token stream), so it is fine-tuned from its labels in
    ``rukh.train.heads`` rather than pretrained here.
    """

    input: Literal["moves"] = "moves"
    model: EncoderConfig | None = None  # overrides the defaults when given
    masking: MaskingConfig = Field(default_factory=MaskingConfig)

    @property
    def default_name(self) -> str:
        return f"encoder-{self.input}"

    def encoder(self) -> EncoderConfig:
        """The encoder config of this run: ``model`` (or the defaults) with ``block`` applied."""
        base = self.model if self.model is not None else EncoderConfig()
        return base.model_copy(update={"block": self.block, "input": self.input})


def masked_step(
    encoder: PositionEncoder,
    body: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Logits and masked-move loss for one batch.

    ``body`` is the encoder or its compiled twin; the head and the loss stay eager because
    ``torch.compile`` wraps ``forward`` and nothing else.
    """
    logits = encoder.mlm_head(body(inputs, attention_mask))
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=MMM_IGNORE_INDEX,
    )
    return logits, loss


@torch.no_grad()
def evaluate_mmm(
    encoder: PositionEncoder,
    body: nn.Module,
    loader: DataLoader[Batch],
    batches: int,
    device: torch.device,
    masking: MaskingConfig,
    vocab: int,
    scheme: Scheme = "moves",
    autocast: Any = None,
) -> tuple[float, float]:
    """Validation loss and accuracy over the hidden positions of at most ``batches`` batches.

    The validation masking is drawn from a generator reseeded here, so every evaluation of a run
    (and of the next run) hides the same positions and the curve compares like with like.
    """
    was_training = body.training
    body.eval()
    generator = masking_generator(masking.seed, device)
    loss_sum = 0.0
    weighted = 0
    hits = 0
    counted = 0
    for index, (x, _) in enumerate(loader):
        if index >= batches:
            break
        x = x.to(device)
        inputs, labels = apply_masking(x, masking, vocab, scheme, generator)
        with autocast if autocast is not None else nullcontext():
            logits, loss = masked_step(encoder, body, inputs, labels, encoder.padding_mask(x))
        scored = labels != MMM_IGNORE_INDEX
        tokens = int(scored.sum())
        if torch.isfinite(loss) and tokens:
            loss_sum += loss.float().item() * tokens
            weighted += tokens
        hits += int((logits.argmax(dim=-1) == labels)[scored].sum())
        counted += tokens
    if was_training:
        body.train()
    return (loss_sum / weighted if weighted else math.nan, hits / counted if counted else 0.0)


def train_mmm(cfg: MmmConfig, resume: Path | None = None, device: str | None = None) -> Path:
    """Pretrain a ``PositionEncoder`` with masked move modeling; return the last checkpoint."""
    cfg.check()
    torch.manual_seed(cfg.seed)
    where = torch.device(device or pick_device())
    tokens_dir = paths.resolve(cfg.tokens_dir)
    model_cfg = cfg.encoder()

    train_set = PackedDataset(tokens_dir / "train", block=cfg.block)
    val_set = PackedDataset(tokens_dir / "val", block=cfg.block)
    if train_set.info.vocab_size != model_cfg.tokens:
        raise ValueError(
            f"{tokens_dir / 'train'} has vocab_size {train_set.info.vocab_size}, "
            f"the model expects {model_cfg.tokens}"
        )
    train_loader = make_loader(train_set, cfg.batch_size, seed=cfg.seed, workers=cfg.workers)
    val_loader = make_loader(
        val_set, cfg.batch_size, seed=cfg.seed, workers=0, shuffle=False, drop_last=False
    )
    if not len(train_loader):
        raise ValueError(f"{tokens_dir / 'train'} has fewer than {cfg.batch_size} windows")

    model = PositionEncoder(model_cfg).to(where)
    vocab = model_cfg.tokens
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    start_step = 0
    best_val = math.inf
    run_id: str | None = None
    if resume is not None:
        start_step, best_val, run_id = load_resume(resume, model, optimizer, where)

    autocast, use_bf16 = autocast_for(cfg, where)
    generator = masking_generator(cfg.masking.seed, where)

    def probe(compiled: nn.Module) -> None:
        """One masked step of the training shape, to force compilation here and not mid-run."""
        warm = torch.full((cfg.batch_size, cfg.block), max(MOVE_CONTROL_IDS) + 1, device=where)
        labels = torch.full_like(warm, MMM_IGNORE_INDEX)
        labels[:, 0] = warm[:, 0]
        _, loss = masked_step(model, compiled, warm, labels)
        loss.backward()

    with autocast:  # compile under the same precision the loop will use
        runnable = maybe_compile(model, cfg.compile, probe=probe)
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
    tags = {"objective": "mmm", "input": cfg.input}
    with start_run(out_dir.name, params, tags=tags, run_id=run_id) as run:
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
            # Weighted by hidden positions, on the device: a micro-batch where the draw hid
            # fewer moves must not weigh as much as one where it hid many.
            loss_sum = torch.zeros((), device=where)
            hidden_tokens = torch.zeros((), device=where)
            for _ in range(cfg.grad_accum):
                x, _ = next(batches)
                x = x.to(where)
                inputs, labels = apply_masking(x, cfg.masking, vocab, cfg.input, generator)
                with autocast:
                    _, loss = masked_step(model, runnable, inputs, labels, model.padding_mask(x))
                (loss / cfg.grad_accum).backward()
                tokens = (labels != MMM_IGNORE_INDEX).sum()
                loss_sum += loss.detach().float() * tokens
                hidden_tokens += tokens
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            optimizer.step()

            done = step + 1
            last = done == cfg.max_steps
            if done % cfg.log_every == 0 or last:
                counted = float(hidden_tokens.item())
                total = float(loss_sum.item()) / counted if counted else math.nan
                elapsed = max(time.perf_counter() - clock, 1e-9)
                steps = min(cfg.log_every, done - start_step)
                log_metrics(
                    {
                        "train/loss": total,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "tokens_per_s": tokens_per_step * steps / elapsed,
                        "masked_tokens_per_s": counted * steps / elapsed,
                    },
                    step=done,
                )
                clock = time.perf_counter()
            if done % cfg.eval_every == 0 or last:
                val_loss, top1 = evaluate_mmm(
                    model,
                    runnable,
                    val_loader,
                    cfg.eval_batches,
                    where,
                    cfg.masking,
                    vocab,
                    cfg.input,
                    autocast if use_bf16 else None,
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
