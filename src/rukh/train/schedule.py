"""Learning-rate schedule: a linear warmup followed by a cosine decay to a floor."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from rukh.train.loop import TrainConfig


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate for ``step`` (0-based).

    It rises linearly from 0 to ``cfg.lr`` over the first ``cfg.warmup`` steps, then follows a
    cosine down to ``cfg.min_lr_ratio * cfg.lr`` at ``cfg.max_steps`` and stays there for any
    later step (so a resumed run past ``max_steps`` never gets a negative or rising rate).
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    floor = cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup:
        return cfg.lr * step / cfg.warmup
    span = cfg.max_steps - cfg.warmup
    if span <= 0:
        return floor
    progress = min(1.0, (step - cfg.warmup) / span)
    return floor + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - floor)
