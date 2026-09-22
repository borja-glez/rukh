"""A reward model: the encoder of M3 with one scalar head, trained on preferences.

The scalar it produces has **no absolute meaning**. Bradley-Terry only ever sees differences --
the loss is ``-log σ(r(better) - r(worse))`` -- so adding a constant to every score changes
nothing, and a reward model that behaves differently when you shift it is broken in a way that
only shows up later. There is a test for exactly that.

Two design choices worth defending.

**The input is the position *after* the move.** A reward over ``(position, move)`` pairs could be
built by feeding both, but the encoder of M3 already takes a FEN and the position after a move is
the move's consequence, which is what a reward is about. It also means the reward model can score
a position that arrived by any route, which is what GRPO needs when it ranks eight candidate moves
against each other.

**The head is unbounded.** A ``tanh`` would keep the scale tidy and would also cap the margin
between a good move and a catastrophic one at the same ceiling, which is the one distinction the
model is being trained to make. The scale is kept in check by the loss itself: pushing the margin
to infinity costs more and more for less and less.

What this module is **not** for: DPO. `DPOTrainer` and `rukh.train.dpo` have an implicit reference
and need no reward model at all. This one is here because it is the piece that makes PPO make
sense, because it can be measured on its own (``docs/acceptance.md`` asks for 75 % on held-out
pairs), and
because a learned reward is the thing GRPO's verifiable rewards are being compared against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rukh.models.encoder import PositionEncoder

__all__ = ["RewardModel", "preference_accuracy", "preference_loss"]


class RewardModel(nn.Module):
    """``PositionEncoder`` plus a scalar head: how good is this position for whoever just moved."""

    def __init__(
        self,
        encoder: PositionEncoder,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = nn.Linear(encoder.cfg.d_model, 1)

    def forward(self, idx: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """``(B,)`` scores. Higher is better for the side that reached this position."""
        pooled = self.encoder.pool(idx, self.pooling, attention_mask)
        return self.head(pooled).squeeze(-1)


def preference_loss(chosen: Tensor, rejected: Tensor) -> Tensor:
    """Bradley-Terry: ``-log σ(r_chosen - r_rejected)``, averaged over the batch.

    Only the difference appears, which is the whole model: preferences say what is better than
    what, never how good anything is on its own.
    """
    return -F.logsigmoid(chosen - rejected).mean()


def preference_accuracy(chosen: Tensor, rejected: Tensor) -> float:
    """Share of pairs the model orders correctly. A tie counts as wrong, not as half right.

    ``docs/acceptance.md`` reads its bar on this number, and a reward model that scores two moves
    exactly
    the same has expressed no preference -- rounding that up to half a point would flatter it.
    """
    with torch.no_grad():
        return float((chosen > rejected).float().mean())
