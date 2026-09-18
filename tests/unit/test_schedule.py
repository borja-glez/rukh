"""Tests for rukh.train.schedule: linear warmup then cosine decay to a floor."""

from __future__ import annotations

import pytest

from rukh.train import TrainConfig, lr_at

pytestmark = pytest.mark.unit

CFG = TrainConfig(lr=6e-4, min_lr_ratio=0.1, warmup=100, max_steps=1000)


def test_warmup_is_linear_and_reaches_the_peak() -> None:
    assert lr_at(0, CFG) == 0.0
    assert lr_at(50, CFG) == pytest.approx(CFG.lr / 2)
    assert lr_at(100, CFG) == pytest.approx(CFG.lr)
    rising = [lr_at(step, CFG) for step in range(101)]
    assert rising == sorted(rising)
    deltas = {round(b - a, 12) for a, b in zip(rising, rising[1:], strict=False)}
    assert len(deltas) == 1


def test_cosine_decays_to_the_floor_and_stays_there() -> None:
    floor = CFG.lr * CFG.min_lr_ratio
    assert lr_at(1000, CFG) == pytest.approx(floor)
    assert lr_at(5000, CFG) == pytest.approx(floor)
    falling = [lr_at(step, CFG) for step in range(100, 1001, 10)]
    assert falling == sorted(falling, reverse=True)
    assert all(value >= floor - 1e-12 for value in falling)
    halfway = lr_at(550, CFG)
    assert halfway == pytest.approx(floor + 0.5 * (CFG.lr - floor))


def test_without_warmup_the_first_step_is_the_peak() -> None:
    cfg = CFG.model_copy(update={"warmup": 0})
    assert lr_at(0, cfg) == pytest.approx(cfg.lr)


def test_degenerate_schedule_returns_the_floor() -> None:
    cfg = CFG.model_copy(update={"warmup": 1000, "max_steps": 1000})
    assert lr_at(1000, cfg) == pytest.approx(cfg.lr * cfg.min_lr_ratio)
    with pytest.raises(ValueError, match="non-negative"):
        lr_at(-1, CFG)


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        TrainConfig(learning_rate=1e-3)  # type: ignore[call-arg]
