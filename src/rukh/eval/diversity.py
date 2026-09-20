"""Opening diversity: how many different games the model is willing to play.

``docs/spec/02`` lists it in the evaluation table -- "entropy of the openings played in 200 games
of its own" -- and it has been ``n/a`` in every report so far. M4 needs it, because the claim the
masters fine-tune makes is precisely a trade: *the Elo goes up and the diversity goes down*. A
number that only measures the first half of that sentence cannot check it.

Two numbers, because they answer different questions and fail in different ways.

**The line entropy** plays ``games`` openings of the model against itself and measures the
entropy of the distribution of distinct opening lines. It is the metric the spec asks for and it
depends on the sampling: read at the deterministic setting the demo plays at, a decoder produces
one single line and the entropy is exactly zero, which is true but says more about the sampler
than about the weights. So it is reported together with the temperature it was read at, like
every other number in this project (D-047).

**The first-move entropy** is analytic: the entropy of the model's own distribution over the
twenty legal first moves, with no sampling at all. It is a property of the weights, it has no
variance between runs, and it is the one to compare across stages. A model that has been
fine-tuned onto a narrow repertoire collapses it whatever the temperature.

Self-play is batched: every game advances one ply at a time, all of them in one forward pass, so
200 games of 12 plies are 12 passes of batch 200 rather than 2 400 passes of batch 1.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import chess
import torch
from pydantic import BaseModel, ConfigDict, Field

from rukh.eval.legality import header
from rukh.infer.sampler import SampleConfig, legal_token_ids, prompt_ids

if TYPE_CHECKING:
    from rukh.models import MoveDecoder
    from rukh.tokenize.uci_vocab import UciTokenizer

__all__ = [
    "DiversityResult",
    "first_move_distribution",
    "first_move_entropy",
    "opening_diversity",
    "shannon_entropy",
]

OPENING_PLIES = 12
"""Six moves a side: long enough to have left the book, short enough that two games sharing it
really are the same opening."""


class DiversityResult(BaseModel):
    """What ``opening_diversity`` measured, with the sampling it was measured at."""

    model_config = ConfigDict(extra="forbid")

    games: int
    plies: int
    distinct_lines: int
    entropy_bits: float
    """Entropy of the distribution over opening lines, in bits."""
    max_entropy_bits: float
    """``log2(games)``: what the entropy would be if every game opened differently."""
    normalised: float = Field(ge=0.0, le=1.0)
    """``entropy_bits / max_entropy_bits``: 1 is a different opening every game, 0 is always
    the same one. It is what makes two runs with different ``games`` comparable."""
    first_move_entropy_bits: float
    """Entropy of the model's distribution over the legal first moves; no sampling involved."""
    temperature: float
    top_k: int | None
    top_lines: list[tuple[str, int]]
    """The most played lines and how often, for the report; a collapsed model shows it here."""


def shannon_entropy(counts: dict[str, int] | Counter[str]) -> float:
    """Entropy of a distribution given as counts, in bits. Empty or single-valued gives 0."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    bits = -sum((n / total) * math.log2(n / total) for n in counts.values() if n > 0)
    # A single outcome gives `-1 * log2(1)`, which is negative zero and renders as "-0.000".
    return 0.0 if bits == 0 else bits


@torch.no_grad()
def first_move_distribution(
    model: MoveDecoder, tok: UciTokenizer, header_elo: int = 1800
) -> dict[str, float]:
    """The model's own probability for each of the twenty legal first moves, renormalised.

    No temperature, no top-k, no sampling: the softmax as the weights produce it. That makes it
    the right instrument for measuring what a style adapter *did* -- the share it moved towards
    ``e2e4`` is a property of the weights, with no seed in it and no variance between runs, which
    self-play openings could not give without playing hundreds of games to average the noise out.
    """
    board = chess.Board()
    device = next(model.parameters()).device
    idx = torch.tensor([header(tok, header_elo, header_elo)], dtype=torch.long, device=device)
    logits = model.next_logits(idx)[0].float()
    legal = legal_token_ids(board, tok)
    kept = logits[torch.tensor(legal, dtype=torch.long, device=device)]
    probs = torch.softmax(kept, dim=-1)
    return {tok.ids[token]: float(p) for token, p in zip(legal, probs.tolist(), strict=True)}


def first_move_entropy(model: MoveDecoder, tok: UciTokenizer, header_elo: int = 1800) -> float:
    """Entropy in bits of the model's own distribution over the legal first moves.

    No temperature, no top-k, no sampling: the softmax as the weights produce it, restricted to
    the twenty legal moves and renormalised. The ceiling is ``log2(20) = 4.32`` bits for a model
    that has no opinion at all; a strong human-imitating model sits well below that, and one
    fine-tuned onto a single repertoire collapses towards zero.
    """
    probs = [p for p in first_move_distribution(model, tok, header_elo).values() if p > 0]
    return float(-sum(p * math.log2(p) for p in probs))


@torch.no_grad()
def opening_diversity(
    model: MoveDecoder,
    tok: UciTokenizer,
    games: int = 200,
    plies: int = OPENING_PLIES,
    sampling: SampleConfig | None = None,
    header_elo: int = 1800,
    seed: int = 42,
) -> DiversityResult:
    """Play ``games`` openings of the model against itself and measure how varied they are."""
    cfg = sampling or SampleConfig()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    start = header(tok, header_elo, header_elo)

    boards = [chess.Board() for _ in range(games)]
    histories = [list(start) for _ in range(games)]
    lines: list[list[str]] = [[] for _ in range(games)]
    alive = list(range(games))

    for _ in range(plies):
        if not alive:
            break
        batch = [prompt_ids(histories[i], model.cfg.block) for i in alive]
        # Every live game is at the same ply, so the batch is rectangular and needs no padding.
        # That is worth an assertion rather than a pad: this model has *learned* position
        # embeddings, so padding on the left would shift every real token to a position it was
        # never trained at, and padding on the right would leave the last column holding a
        # `<pad>` for the shorter rows. Both read the wrong distribution, silently.
        width = len(batch[0])
        if any(len(ids) != width for ids in batch):
            raise RuntimeError("self-play games fell out of step; the batch cannot be padded")
        padded = torch.tensor(batch, dtype=torch.long, device=device)
        logits = model(padded)[0][:, -1, :].float()

        still: list[int] = []
        for row, index in enumerate(alive):
            board = boards[index]
            legal = legal_token_ids(board, tok)
            if not legal:
                continue
            filtered = _mask(logits[row], legal, cfg)
            probs = torch.softmax(filtered, dim=-1)
            if cfg.temperature == 0:
                token = int(torch.argmax(filtered))
            else:
                token = int(torch.multinomial(probs, 1, generator=generator))
            uci = tok.ids[token]
            move = chess.Move.from_uci(uci)
            board.push(move)
            histories[index].append(token)
            lines[index].append(uci)
            if not board.is_game_over():
                still.append(index)
        alive = still

    counts = Counter(" ".join(line) for line in lines if line)
    entropy = shannon_entropy(counts)
    ceiling = math.log2(games) if games > 1 else 0.0
    return DiversityResult(
        games=games,
        plies=plies,
        distinct_lines=len(counts),
        entropy_bits=entropy,
        max_entropy_bits=ceiling,
        normalised=min(entropy / ceiling, 1.0) if ceiling > 0 else 0.0,
        first_move_entropy_bits=first_move_entropy(model, tok, header_elo),
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_lines=counts.most_common(5),
    )


def _mask(logits: torch.Tensor, legal: list[int], cfg: SampleConfig) -> torch.Tensor:
    """The legality mask, the temperature and the top-k of ``SampleConfig``, on one row.

    Always masked, whatever ``cfg.mask_illegal`` says: this measures which *openings* the model
    chooses, and an illegal token is not an opening. Legality has its own metric.
    """
    out = logits.clone()
    keep = torch.zeros_like(out, dtype=torch.bool)
    keep[torch.tensor(legal, dtype=torch.long, device=out.device)] = True
    out = out.masked_fill(~keep, float("-inf"))
    if cfg.temperature > 0:
        out = out / cfg.temperature
    if cfg.top_k is not None:
        k = min(cfg.top_k, len(legal))
        threshold = torch.topk(out, k).values[-1]
        out = out.masked_fill(out < threshold, float("-inf"))
    return out
