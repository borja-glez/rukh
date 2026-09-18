"""Next-move accuracy against the human move, overall and per Elo band.

The raw logits are used, not the masked ones: this metric asks "would the model have played
what the human played", and hiding the illegal tokens would flatter it. Bands are 200 Elo wide
starting at 1800 (the floor of ``rukh-games-1800``), with ``<1800`` and ``2600+`` catch-alls,
and a position is filed under the rating of the side to move.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from pydantic import BaseModel, ConfigDict

from rukh.eval.legality import Position
from rukh.infer import prompt_ids
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

BAND_START = 1800
BAND_WIDTH = 200
BAND_TOP = 2600
TOP_K = 3


def elo_band(elo: int) -> str:
    """Name of the 200-Elo band ``elo`` falls in."""
    if elo < BAND_START:
        return f"<{BAND_START}"
    if elo >= BAND_TOP:
        return f"{BAND_TOP}+"
    low = (elo - BAND_START) // BAND_WIDTH * BAND_WIDTH + BAND_START
    return f"{low}-{low + BAND_WIDTH}"


class BandAccuracy(BaseModel):
    """Accuracy inside one Elo band."""

    model_config = ConfigDict(extra="forbid")

    band: str
    positions: int
    top1: float
    top3: float


class AccuracyResult(BaseModel):
    """Top-1 and top-3 agreement with the human move."""

    model_config = ConfigDict(extra="forbid")

    positions: int
    top1: float
    top3: float
    bands: list[BandAccuracy]


def _ranked(model: MoveDecoder, history: list[int], k: int) -> list[int]:
    """Ids of the ``k`` highest logits at the next step (prompt cropped header-first)."""
    device = next(model.parameters()).device
    ids = prompt_ids(history, model.cfg.block)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model.next_logits(idx)[0]
    return [int(i) for i in torch.topk(logits, min(k, logits.numel())).indices]


def accuracy(
    model: MoveDecoder,
    tok: UciTokenizer,
    positions: Sequence[Position],
    top_k: int = TOP_K,
) -> AccuracyResult:
    """Compare the model's ``top_k`` tokens with the move the human actually played."""
    hits1: dict[str, int] = {}
    hits3: dict[str, int] = {}
    counts: dict[str, int] = {}
    for position in positions:
        if position.target is None or position.target not in tok.vocab:
            continue
        band = elo_band(position.elo)
        counts[band] = counts.get(band, 0) + 1
        target = tok.vocab[position.target]
        ranked = _ranked(model, position.history, top_k)
        hits1[band] = hits1.get(band, 0) + int(bool(ranked) and ranked[0] == target)
        hits3[band] = hits3.get(band, 0) + int(target in ranked)
    total = sum(counts.values())
    bands = [
        BandAccuracy(
            band=band,
            positions=counts[band],
            top1=hits1.get(band, 0) / counts[band],
            top3=hits3.get(band, 0) / counts[band],
        )
        for band in sorted(counts)
    ]
    return AccuracyResult(
        positions=total,
        top1=sum(hits1.values()) / total if total else 0.0,
        top3=sum(hits3.values()) / total if total else 0.0,
        bands=bands,
    )
