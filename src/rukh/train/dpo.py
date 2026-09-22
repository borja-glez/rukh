"""Direct Preference Optimization on the decoder, over engine-scored move pairs.

Every other lever tried so far teaches the model to *imitate*: more games, better games, a header
saying how strong the players were. Imitation is capped by the corpus, and D-069 measured that cap
-- conditioning on 2600 does not make the model play better, it makes it play like a 2600 losing
to tactics. DPO changes the objective. Each example says "in this position `chosen` is at least
``min_delta_cp`` better than `rejected`", which is information no human game contains, and the
loss pushes probability from one to the other instead of towards the average continuation.

Our completion is a single token -- one move -- so the usual sequence-level DPO collapses to the
log-probability of two tokens at one position, which is why this file is short.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field

from rukh.config import BaseConfig
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer, elo_token

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


class DpoConfig(BaseConfig):
    """Where the pairs and the starting weights are, and the two knobs that matter."""

    pairs: str = "data/pairs/dpo-prompts.parquet"
    max_pairs: int | None = Field(default=None, ge=1)
    """Subsample the pairs to this many, with ``seed``, before the split. Two arms compared on
    two pair sources must see the same amount of data, or the comparison measures the amount."""
    checkpoint: str = ""
    out_dir: str = "checkpoints"
    run_name: str = "dpo"
    beta: float = Field(default=0.1, gt=0)
    """How hard the loss pushes. Above ~0.5 the policy leaves the reference and legality goes."""
    lr: float = Field(default=5e-6, gt=0)
    nll_weight: float = Field(default=0.0, ge=0)
    """Optional anchor on the chosen move. Pure DPO can raise the margin while the model forgets
    how to play at all; a little next-token loss keeps it honest."""
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=1, ge=1)
    block: int = Field(default=200, ge=8)
    val_fraction: float = Field(default=0.1, ge=0.0, lt=0.5)
    seed: int = 42
    device: str | None = None


class PreferencePair(NamedTuple):
    """One encoded pair, carrying the game it came from so the split can go by game."""

    game_id: str
    ids: list[int]
    chosen: int
    rejected: int


class PreferenceBatch:
    """Padded prompts plus the index of the position that predicts the move."""

    def __init__(
        self, tokens: torch.Tensor, last: torch.Tensor, chosen: torch.Tensor, rejected: torch.Tensor
    ) -> None:
        self.tokens, self.last, self.chosen, self.rejected = tokens, last, chosen, rejected

    def to(self, device: torch.device) -> PreferenceBatch:
        return PreferenceBatch(
            self.tokens.to(device),
            self.last.to(device),
            self.chosen.to(device),
            self.rejected.to(device),
        )

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


def encode_pairs(frame: object, tok: UciTokenizer, block: int) -> list[PreferencePair]:
    """One :class:`PreferencePair` per usable row.

    A row is dropped when either move is outside the fixed vocabulary (a promotion spelling the
    enumeration does not carry) or when the prompt would not fit in ``block``: silently
    truncating a prefix would hand the model a position that never happened.

    ``game_id`` travels with the pair and is never read by the loss: it exists so that
    :func:`split_pairs` can hold out whole games (D-070).
    """
    rows: list[PreferencePair] = []
    for game_id, prefix, chosen, rejected, white, black in zip(
        frame["game_id"],
        frame["prefix"],  # type: ignore[index]
        frame["chosen"],
        frame["rejected"],  # type: ignore[index]
        frame["white_elo"],
        frame["black_elo"],
        strict=True,  # type: ignore[index]
    ):
        chosen_id = tok.vocab.get(str(chosen))
        rejected_id = tok.vocab.get(str(rejected))
        if chosen_id is None or rejected_id is None or chosen_id == rejected_id:
            continue
        ids = [
            tok.bos_id,
            tok.vocab[elo_token(int(white), "w")],
            tok.vocab[elo_token(int(black), "b")],
        ]
        moves = [tok.vocab.get(m) for m in str(prefix).split()]
        if any(m is None for m in moves):
            continue
        ids += [int(m) for m in moves]  # type: ignore[arg-type]
        if len(ids) > block:
            continue
        rows.append(PreferencePair(str(game_id), ids, chosen_id, rejected_id))
    return rows


def batches(
    rows: list[PreferencePair], size: int, pad_id: int, shuffle: bool, seed: int
) -> Iterator[PreferenceBatch]:
    """Pad each batch to its own longest prompt and remember where each prompt ends."""
    order = np.arange(len(rows))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), size):
        chunk = [rows[int(i)] for i in order[start : start + size]]
        if not chunk:
            continue
        width = max(len(row.ids) for row in chunk)
        tokens = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        last = torch.empty(len(chunk), dtype=torch.long)
        for i, row in enumerate(chunk):
            tokens[i, : len(row.ids)] = torch.tensor(row.ids, dtype=torch.long)
            last[i] = len(row.ids) - 1
        yield PreferenceBatch(
            tokens,
            last,
            torch.tensor([row.chosen for row in chunk], dtype=torch.long),
            torch.tensor([row.rejected for row in chunk], dtype=torch.long),
        )


def move_logprobs(model: MoveDecoder, batch: PreferenceBatch) -> torch.Tensor:
    """Log-probabilities over the vocabulary at the position that predicts the next move."""
    logits, _ = model(batch.tokens)
    picked = logits[torch.arange(logits.shape[0], device=logits.device), batch.last]
    return F.log_softmax(picked.float(), dim=-1)


def dpo_loss(
    policy: torch.Tensor, reference: torch.Tensor, batch: PreferenceBatch, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The DPO objective for a one-token completion, plus the margin and the accuracy.

    ``margin`` is ``rewards/margins`` in the usual reporting: how much more the policy prefers the
    better move than the reference did. ``accuracy`` is the share of pairs it orders correctly,
    which is the number to watch -- a margin can grow while the ordering stays wrong.
    """
    index = torch.arange(policy.shape[0], device=policy.device)
    chosen = policy[index, batch.chosen] - reference[index, batch.chosen]
    rejected = policy[index, batch.rejected] - reference[index, batch.rejected]
    margin = chosen - rejected
    loss = -F.logsigmoid(beta * margin).mean()
    accuracy = (policy[index, batch.chosen] > policy[index, batch.rejected]).float().mean()
    return loss, margin.mean(), accuracy


def evaluate(
    policy: MoveDecoder,
    reference: MoveDecoder,
    rows: list[PreferencePair],
    cfg: DpoConfig,
    device: torch.device,
    pad_id: int,
) -> tuple[float, float, float]:
    """Mean loss, margin and pair accuracy over held-out pairs."""
    policy.eval()
    totals = [0.0, 0.0, 0.0]
    seen = 0
    with torch.no_grad():
        for batch in batches(rows, cfg.batch_size, pad_id, shuffle=False, seed=cfg.seed):
            batch = batch.to(device)
            loss, margin, accuracy = dpo_loss(
                move_logprobs(policy, batch), move_logprobs(reference, batch), batch, cfg.beta
            )
            n = len(batch)
            totals[0] += float(loss) * n
            totals[1] += float(margin) * n
            totals[2] += float(accuracy) * n
            seen += n
    policy.train()
    if not seen:
        return math.nan, math.nan, math.nan
    return totals[0] / seen, totals[1] / seen, totals[2] / seen


def split_pairs(
    rows: list[PreferencePair], val_fraction: float, seed: int
) -> tuple[list[PreferencePair], list[PreferencePair]]:
    """Hold out whole **games**, never single pairs.

    One game contributes several pairs a few plies apart, and they share a prefix almost to the
    end: a pair of the same game on the other side of the split is a position the policy has
    already trained on, so the accuracy printed for validation would be measuring memorisation.
    The reward model splits the same way and for the same reason (``train.reward.split_examples``).
    """
    from rukh.data.labels import game_split

    train: list[PreferencePair] = []
    val: list[PreferencePair] = []
    for row in rows:
        (val if game_split(row.game_id, val_fraction, seed) == "val" else train).append(row)
    return train, val


def limit_pairs(frame: Any, cfg: DpoConfig) -> Any:
    """At most ``cfg.max_pairs`` rows, drawn with ``cfg.seed`` from the whole file.

    A sample and never the head: the pairs are balanced by phase in blocks, so the first rows
    of the file are two thirds openings against a third in the whole (D-114). With no limit the
    frame comes back untouched.
    """
    if cfg.max_pairs is None or frame.height <= cfg.max_pairs:
        return frame
    LOGGER.info("dpo: %d pairs sampled out of %d", cfg.max_pairs, frame.height)
    return frame.sample(n=cfg.max_pairs, seed=cfg.seed, shuffle=True)


def train(cfg: DpoConfig) -> Path:
    """Align a decoder on the preference pairs and return the checkpoint it wrote."""
    import copy

    import polars as pl

    from rukh import paths
    from rukh.train.checkpoint import load_model, save_checkpoint
    from rukh.train.common import pick_device

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device or pick_device())
    tok = UciTokenizer()

    policy, payload = load_model(paths.resolve(cfg.checkpoint), device)
    policy = policy.to(device).train()
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    frame = limit_pairs(pl.read_parquet(paths.resolve(cfg.pairs).as_posix()), cfg)
    rows = encode_pairs(frame, tok, cfg.block)
    if not rows:
        raise ValueError(f"{cfg.pairs} produced no usable pairs")
    train_rows, val_rows = split_pairs(rows, cfg.val_fraction, cfg.seed)
    LOGGER.info("dpo: %d training pairs, %d validation", len(train_rows), len(val_rows))

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    out_dir = paths.resolve(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    before = evaluate(policy, reference, val_rows, cfg, device, tok.pad_id)
    LOGGER.info("before: loss %.4f | margin %.4f | accuracy %.4f", *before)
    step = 0
    for epoch in range(cfg.epochs):
        for batch in batches(
            train_rows, cfg.batch_size, tok.pad_id, shuffle=True, seed=cfg.seed + epoch
        ):
            batch = batch.to(device)
            policy_logprobs = move_logprobs(policy, batch)
            with torch.no_grad():
                reference_logprobs = move_logprobs(reference, batch)
            loss, margin, accuracy = dpo_loss(policy_logprobs, reference_logprobs, batch, cfg.beta)
            if cfg.nll_weight:
                index = torch.arange(policy_logprobs.shape[0], device=device)
                loss = loss - cfg.nll_weight * policy_logprobs[index, batch.chosen].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 50 == 0:
                LOGGER.info(
                    "step %d | loss %.4f | margin %.4f | accuracy %.4f",
                    step,
                    float(loss),
                    float(margin),
                    float(accuracy),
                )
    after = evaluate(policy, reference, val_rows, cfg, device, tok.pad_id)
    LOGGER.info("after: loss %.4f | margin %.4f | accuracy %.4f", *after)

    final = out_dir / "dpo.pt"
    save_checkpoint(
        final,
        step=step,
        model=policy,
        optimizer=None,
        cfg={
            **cfg.model_dump(mode="json"),
            "val_before": dict(zip(("loss", "margin", "accuracy"), before, strict=True)),
            "val_after": dict(zip(("loss", "margin", "accuracy"), after, strict=True)),
        },
        model_cfg=payload["model_cfg"],
        vocab_hash=payload.get("vocab_hash"),
        data_manifest_sha=payload.get("data_manifest_sha"),
    )
    return final
