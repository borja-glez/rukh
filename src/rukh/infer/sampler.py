"""Sampling one move from a ``MoveDecoder``, with or without the legality mask.

The mask is the honest part of the demo: with ``mask_illegal`` the logits of every token that
is not a legal move become ``-inf``, so an illegal move is impossible; without it the raw token
is returned as it came out of the network and reported with ``legal=False``, which is how the
legality rate (the "understanding" metric of ``docs/spec/02``) is measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import chess
import torch
from pydantic import model_validator

from rukh.config import BaseConfig
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

TOP_N = 5
HEADER_TOKENS = 3
"""``<bos> <wXXXX> <bXXXX>``: the control prefix every training sequence starts with."""


class SampleConfig(BaseConfig):
    """How a move is drawn: temperature, top-k truncation and the legality mask."""

    temperature: float = 0.6
    top_k: int | None = 20
    mask_illegal: bool = True
    seed: int | None = None

    @model_validator(mode="after")
    def _check(self) -> SampleConfig:
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        return self

    def generator(self, device: torch.device | str | None = None) -> torch.Generator | None:
        """A seeded generator on ``device``, or None when the run is not reproducible.

        The device matters: ``torch.multinomial`` refuses a CPU generator when the
        probabilities live on CUDA, so the generator has to be built where the model is.
        ``model_generator`` does that for a given model.
        """
        if self.seed is None:
            return None
        return torch.Generator(device=device or "cpu").manual_seed(self.seed)


def model_generator(model: MoveDecoder, cfg: SampleConfig) -> torch.Generator | None:
    """The sampling generator of ``cfg`` built on the device the model's weights live on."""
    return cfg.generator(next(model.parameters()).device)


def prompt_ids(history: Sequence[int], block: int) -> list[int]:
    """Crop ``history`` to ``block`` ids **keeping the header**: ``header + moves[-(block - 3):]``.

    A plain left crop (``history[-block:]``) silently drops ``<bos> <wXXXX> <bXXXX>`` as soon as a
    game is longer than the context, so the model loses the Elo conditioning it was trained with
    and every learned position is shifted by three. Keeping the header costs three move tokens and
    keeps the prompt shaped like the training data.
    """
    ids = list(history)
    if len(ids) <= block:
        return ids
    if block <= HEADER_TOKENS:
        return ids[:block]
    return ids[:HEADER_TOKENS] + ids[len(ids) - (block - HEADER_TOKENS) :]


def legal_token_ids(board: chess.Board, tok: UciTokenizer) -> list[int]:
    """Sorted vocabulary ids of every legal move in ``board``.

    Castling is the king's two-square move (``e1g1``), the same string the tokenizer enumerates;
    a legal move that is somehow missing from the vocabulary is skipped rather than guessed.
    """
    ids = {tok.vocab[uci] for move in board.legal_moves if (uci := move.uci()) in tok.vocab}
    return sorted(ids)


def _filtered_logits(logits: torch.Tensor, legal: list[int], cfg: SampleConfig) -> torch.Tensor:
    """Apply the legality mask, the temperature and the top-k truncation, in that order."""
    out = logits.float()
    if cfg.mask_illegal:
        keep = torch.zeros_like(out, dtype=torch.bool)
        keep[torch.tensor(legal, dtype=torch.long, device=out.device)] = True
        out = out.masked_fill(~keep, float("-inf"))
    if cfg.temperature > 0:
        out = out / cfg.temperature
    if cfg.top_k is not None:
        k = min(cfg.top_k, out.numel())
        threshold = torch.topk(out, k).values[-1]
        out = out.masked_fill(out < threshold, float("-inf"))
    return out


def pick_move(
    model: MoveDecoder,
    tok: UciTokenizer,
    board: chess.Board,
    history_ids: list[int],
    cfg: SampleConfig,
    generator: torch.Generator | None = None,
) -> tuple[chess.Move | None, dict[str, Any]]:
    """Draw the next move token and report what the network proposed.

    Returns the move (``None`` when the proposed token is not a legal move) and a report with
    ``legal``, ``raw_token``, ``top5`` (token, probability of the distribution actually sampled)
    and ``masked``. ``generator`` lets a whole game advance one seeded stream; build it with
    ``model_generator`` so it lives on the same device as the weights.

    The history is cropped with ``prompt_ids``, so a game longer than the context keeps its
    ``<bos>`` and Elo tokens instead of being cut off the left edge.
    """
    legal = legal_token_ids(board, tok)
    report: dict[str, Any] = {
        "legal": False,
        "raw_token": None,
        "top5": [],
        "masked": cfg.mask_illegal,
    }
    if cfg.mask_illegal and not legal:
        return None, report

    device = next(model.parameters()).device
    ids = prompt_ids(history_ids, model.cfg.block)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits = _filtered_logits(model.next_logits(idx)[0], legal, cfg)
    probs = torch.softmax(logits, dim=-1)
    if cfg.temperature == 0:
        token_id = int(torch.argmax(logits))
    else:
        token_id = int(torch.multinomial(probs, 1, generator=generator))

    top = torch.topk(probs, min(TOP_N, probs.numel()))
    report["top5"] = [
        (tok.ids[int(i)], float(p)) for p, i in zip(top.values, top.indices, strict=True) if p > 0
    ]
    report["raw_token"] = tok.ids[token_id]
    move = _as_move(report["raw_token"])
    if move is not None and board.is_legal(move):
        report["legal"] = True
        return move, report
    return None, report


def _as_move(token: str) -> chess.Move | None:
    """A move token as a ``chess.Move``; special tokens such as ``<eos>`` give None."""
    try:
        return chess.Move.from_uci(token)
    except ValueError:
        return None
