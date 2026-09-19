"""Tests for rukh.data.labels: the blunder sign, the split by game and the nested subsets."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from rukh.data.labels import (
    LABELS_FILE,
    RESULT_CLASSES,
    VALUE_EDGES,
    LabelsConfig,
    build_labels,
    counts,
    game_split,
    label_subsets,
    run,
)

pytestmark = pytest.mark.unit

WHITE_TO_MOVE = "8/8/8/8/8/8/8/8 w - -"
BLACK_TO_MOVE = "8/8/8/8/8/8/8/8 b - -"


def position(
    game_id: int,
    ply: int,
    cp: int | None = 0,
    mate: int | None = None,
    result: str = "1-0",
) -> dict[str, object]:
    """One row of ``positions-eval.parquet``; the side to move follows the ply, as in a game."""
    return {
        # A ply-N position is the one *after* the Nth half-move, so White is to move on even
        # plies: the move that led to an even-ply position was Black's.
        "fen": WHITE_TO_MOVE if ply % 2 == 0 else BLACK_TO_MOVE,
        "game_id": game_id,
        "ply": ply,
        "last_move": "e2e4",
        "result": result,
        "phase": "middlegame",
        "cp": cp,
        "mate": mate,
        "n_seen": 1,
        "best_move": "e2e4",
        "depth": 20,
    }


def rows_to_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def toy_frame() -> pl.DataFrame:
    """Two games whose evaluations swing by the same 300 cp, once for each side."""
    return rows_to_frame(
        [
            # Black moved into ply 2 and handed White 300 centipawns: a blunder.
            position(10, 1, cp=0),
            position(10, 2, cp=300),
            # White moved into ply 3 and gained the same 300: the very same swing, not a blunder.
            position(11, 2, cp=0),
            position(11, 3, cp=300),
        ]
    )


def test_a_blunder_is_measured_from_the_side_that_moved() -> None:
    frame = build_labels(LabelsConfig(), toy_frame()).sort(["game_id", "ply"])
    by_key = {(int(r["game_id"]), int(r["ply"])): r for r in frame.to_dicts()}

    black_blundered = by_key[(10, 2)]
    assert black_blundered["loss_cp"] == 300
    assert black_blundered["blunder"] == 1

    white_gained = by_key[(11, 3)]
    assert white_gained["loss_cp"] == -300  # the same swing, the other side to move
    assert white_gained["blunder"] == 0


def test_a_position_without_its_predecessor_has_no_blunder_label() -> None:
    frame = build_labels(LabelsConfig(), toy_frame())
    first = frame.filter((pl.col("game_id") == 10) & (pl.col("ply") == 1)).to_dicts()[0]
    assert first["blunder"] is None and first["loss_cp"] is None
    assert frame.filter(pl.col("blunder").is_null()).height == 2  # one per game
    # A null is not a zero: a position we cannot judge must never train the head as "fine".
    assert frame.drop_nulls("blunder").height == 2


def test_the_blunder_threshold_is_configurable() -> None:
    rows = rows_to_frame([position(1, 2, cp=0), position(1, 3, cp=-120)])
    strict = build_labels(LabelsConfig(blunder_cp=100), rows)
    lenient = build_labels(LabelsConfig(blunder_cp=200), rows)
    assert strict.filter(pl.col("ply") == 3)["blunder"].item() == 1
    assert lenient.filter(pl.col("ply") == 3)["blunder"].item() == 0


def test_a_mate_counts_as_a_score_and_saturates_the_value() -> None:
    rows = rows_to_frame([position(2, 2, cp=None, mate=3), position(2, 3, cp=None, mate=-1)])
    frame = build_labels(LabelsConfig(), rows).sort("ply")
    assert frame["cp"].to_list() == [10_000 - 3, -10_000 + 1]
    assert math.isclose(frame["value"][0], 1.0, abs_tol=1e-6)
    assert math.isclose(frame["value"][1], -1.0, abs_tol=1e-6)
    assert frame["value_bucket"].to_list() == [4, 0]


def test_value_is_the_bounded_tanh_of_the_centipawns_from_whites_side() -> None:
    rows = rows_to_frame([position(3, ply, cp=cp) for ply, cp in enumerate([-400, 0, 400], 2)])
    frame = build_labels(LabelsConfig(), rows).sort("ply")
    assert [round(v, 4) for v in frame["value"].to_list()] == [
        round(math.tanh(-1.0), 4),
        0.0,
        round(math.tanh(1.0), 4),
    ]
    assert VALUE_EDGES == (-200, -50, 50, 200)
    buckets = build_labels(
        LabelsConfig(),
        rows_to_frame([position(4, p, cp=cp) for p, cp in enumerate([-300, -100, 0, 100, 300], 2)]),
    ).sort("ply")
    assert buckets["value_bucket"].to_list() == [0, 1, 2, 3, 4]


def test_the_result_class_follows_the_game_it_came_from() -> None:
    rows = rows_to_frame(
        [position(5, 2, result=r) for r in ("1-0", "0-1", "1/2-1/2")] + [position(5, 3, result="*")]
    )
    frame = build_labels(LabelsConfig(), rows)
    assert RESULT_CLASSES == {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
    assert sorted(frame["result_class"].to_list()) == [0, 1, 2]  # "*" is dropped, not guessed


def test_the_split_is_by_game_and_never_shares_a_game_id() -> None:
    rows = rows_to_frame([position(game, ply) for game in range(200) for ply in (2, 3, 4)])
    frame = build_labels(LabelsConfig(val_fraction=0.25), rows)
    train = set(frame.filter(pl.col("split") == "train")["game_id"].to_list())
    val = set(frame.filter(pl.col("split") == "val")["game_id"].to_list())
    assert train and val
    assert not train & val  # the whole point: no game is on both sides
    assert len(train) + len(val) == 200
    assert 0.15 < len(val) / 200 < 0.35
    # Every position of a game goes with it.
    for game_id, split in frame.select("game_id", "split").unique().group_by("game_id"):
        assert len(split) == 1, game_id


def test_the_split_is_stable_and_depends_on_the_seed() -> None:
    assert game_split(17, 0.1, 42) == game_split(17, 0.1, 42)
    ids = range(500)
    one = [game_split(i, 0.2, 1) for i in ids]
    two = [game_split(i, 0.2, 2) for i in ids]
    assert one != two
    assert all(value in ("train", "val") for value in one)


def test_the_label_curve_uses_nested_subsets() -> None:
    rows = rows_to_frame([position(game, ply) for game in range(100) for ply in (2, 3)])
    frame = build_labels(LabelsConfig(), rows)
    subsets = label_subsets(frame, [0.1, 0.25, 0.5, 1.0])
    keys = {
        fraction: {(r["game_id"], r["ply"]) for r in subset.to_dicts()}
        for fraction, subset in subsets.items()
    }
    assert keys[0.1] < keys[0.25] < keys[0.5] < keys[1.0]
    assert len(keys[1.0]) == frame.height
    for fraction in (0.1, 0.25, 0.5):
        assert abs(len(keys[fraction]) / frame.height - fraction) < 0.01
    assert label_subsets(frame, [0.5, 0.5]).keys() == {0.5}
    with pytest.raises(ValueError, match="every fraction"):
        label_subsets(frame, [0.0])


def test_run_writes_the_parquet_and_a_manifest(rukh_home: Path) -> None:
    source = rukh_home / "evals" / "positions-eval.parquet"
    source.parent.mkdir(parents=True)
    toy_frame().write_parquet(source)
    cfg = LabelsConfig(positions_eval="evals/positions-eval.parquet", out_dir="labels")
    manifest = run(cfg)
    written = rukh_home / "labels" / LABELS_FILE
    assert written.is_file()
    assert manifest.counts["positions"] == 4
    assert manifest.counts["blunder_labelled"] == 2
    assert manifest.counts["blunders"] == 1
    assert manifest.filters["split"]["by"] == "game_id"
    assert (rukh_home / "labels" / "manifest.json").is_file()
    assert counts(pl.read_parquet(written))["positions"] == 4
