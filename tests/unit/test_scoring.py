"""Tests for rukh.data.scoring: the mate convention and the shared line ordering."""

from __future__ import annotations

import pytest

from rukh.data.scoring import MATE_SCORE, line_rank_key, mover_sign, score_mover, score_white

pytestmark = pytest.mark.unit


def test_score_white_prefers_the_shorter_mate() -> None:
    assert score_white(120, None) == 120
    assert score_white(None, 1) == MATE_SCORE - 1
    assert score_white(None, 5) == MATE_SCORE - 5
    assert score_white(None, -1) == -MATE_SCORE + 1
    assert score_white(None, -5) == -MATE_SCORE + 5
    assert score_white(999, 2) == MATE_SCORE - 2  # mate wins over cp
    assert score_white(None, None) is None


def test_score_mover_flips_for_black() -> None:
    assert mover_sign("w") == 1 and mover_sign("b") == -1
    assert score_mover(150, None, "w") == 150
    assert score_mover(150, None, "b") == -150
    assert score_mover(None, 2, "w") == MATE_SCORE - 2
    assert score_mover(None, -2, "b") == MATE_SCORE - 2
    assert score_mover(None, None, "b") is None


def test_line_rank_key_orders_mate_then_depth_then_cp() -> None:
    lines = [
        ("deep_cp", dict(cp=400, mate=None, depth=40)),
        ("shallow_mate", dict(cp=None, mate=4, depth=12)),
        ("shallow_cp", dict(cp=900, mate=None, depth=12)),
        ("faster_mate", dict(cp=None, mate=2, depth=12)),
        ("getting_mated", dict(cp=None, mate=-3, depth=40)),
    ]
    order = [
        name
        for name, kw in sorted(
            lines, key=lambda item: line_rank_key(move=item[0], turn="w", **item[1])
        )
    ]
    assert order[:2] == ["faster_mate", "shallow_mate"]
    # Depth outranks the score, so the two depth-40 lines come before the depth-12 one.
    assert order[2:] == ["deep_cp", "getting_mated", "shallow_cp"]
    # For Black the signs flip: a mate for Black comes first.
    black = sorted(lines, key=lambda item: line_rank_key(move=item[0], turn="b", **item[1]))
    assert black[0][0] == "getting_mated"


def test_line_rank_key_rejects_an_unscored_line() -> None:
    with pytest.raises(ValueError):
        line_rank_key(None, None, 30, "e2e4", "w")
