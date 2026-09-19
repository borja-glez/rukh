"""The three supervised heads of M3, all of them reading one vector: ``PositionEncoder.pool``.

The heads are deliberately tiny — one linear layer each — because that is what makes the
experiment mean something. If a linear probe on the pooled representation can say how good a
position is, whether the last move threw the game away and who is going to win, then the
representation already contains those facts and masked move modeling put them there. A deep
head would be able to learn them by itself and would tell us nothing about the encoder.

* ``ValueHead``: one scalar through ``tanh``, matching the ``tanh(cp / 400)`` label, so the
  output is bounded and a mate cannot dominate the loss.
* ``BlunderHead``: one logit, because the label is missing for a good part of the rows (a
  position whose predecessor is not in the table) and a masked binary cross entropy is the
  honest way to skip them.
* ``ResultHead``: three classes, White / draw / Black.

``MultiHead`` trains them together over a shared encoder, with a weight per head: the three
tasks have different scales (an MSE in ``[0, 4]``, two cross entropies) and different amounts of
data, so "just add them up" would silently make one of them the only one that matters.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rukh.config import BaseConfig
from rukh.models.encoder import PositionEncoder

HEADS = ("value", "blunder", "result")
RESULT_CLASSES = 3


class HeadWeights(BaseConfig):
    """Weight of each head in the joint loss; ``0`` switches a head off without removing it."""

    value: float = 1.0
    blunder: float = 1.0
    result: float = 0.5


class ValueHead(nn.Module):
    """How good is this position for White: a scalar in ``(-1, 1)``."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, pooled: Tensor) -> Tensor:
        return torch.tanh(self.proj(pooled)).squeeze(-1)


class BlunderHead(nn.Module):
    """Did the move that led here throw the game away: one logit (not a probability)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.proj(pooled).squeeze(-1)


class ResultHead(nn.Module):
    """How the game ended: logits over White / draw / Black."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, RESULT_CLASSES)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.proj(pooled)


class MultiHead(nn.Module):
    """An encoder plus the three heads, trained on one pooled representation."""

    def __init__(
        self,
        encoder: PositionEncoder,
        weights: HeadWeights | None = None,
        pooling: Literal["cls", "mean"] = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.weights = weights or HeadWeights()
        self.pooling = pooling
        d_model = encoder.cfg.d_model
        self.value = ValueHead(d_model)
        self.blunder = BlunderHead(d_model)
        self.result = ResultHead(d_model)

    def pooled(self, idx: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """The position's representation, ``(B, d_model)``."""
        return self.encoder.pool(idx, self.pooling, attention_mask)

    def forward(self, idx: Tensor, attention_mask: Tensor | None = None) -> dict[str, Tensor]:
        """``{"value": (B,), "blunder": (B,) logits, "result": (B, 3) logits}``."""
        return self.from_pooled(self.pooled(idx, attention_mask))

    def from_pooled(self, pooled: Tensor) -> dict[str, Tensor]:
        """The three heads' outputs for an already pooled batch."""
        return {
            "value": self.value(pooled),
            "blunder": self.blunder(pooled),
            "result": self.result(pooled),
        }

    def loss(
        self, outputs: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """``(total, per head)``; a head with no labelled row in the batch contributes zero.

        ``targets`` carries ``value`` (float), ``blunder`` (0/1 float) with its ``blunder_mask``
        (the rows that have a label at all) and ``result`` (class index).
        """
        zero = torch.zeros((), device=outputs["value"].device, dtype=outputs["value"].dtype)
        parts = {
            "value": F.mse_loss(outputs["value"], targets["value"].to(outputs["value"].dtype)),
            "blunder": zero,
            "result": F.cross_entropy(outputs["result"], targets["result"].long()),
        }
        mask = targets["blunder_mask"].bool()
        if bool(mask.any()):
            parts["blunder"] = F.binary_cross_entropy_with_logits(
                outputs["blunder"][mask], targets["blunder"][mask].to(outputs["blunder"].dtype)
            )
        total = sum(
            (getattr(self.weights, name) * parts[name] for name in HEADS),
            start=torch.zeros_like(zero),
        )
        return total, parts

    def head_parameters(self) -> list[nn.Parameter]:
        """Every parameter of the three heads, the ones a ``probe`` run is allowed to move."""
        return [
            param for head in (self.value, self.blunder, self.result) for param in head.parameters()
        ]
