"""The supervised dataset of M3: value, blunder and result labels from the P1 evaluations.

Everything here comes out of ``data/evals/positions-eval.parquet``, the table P1 built by
crossing the sampled positions with ``Lichess/chess-position-evaluations`` (see
``rukh.data.evals``): one row per distinct four-field FEN, with the game it first came from
(``game_id``, ``ply``, ``last_move``, ``result``) and Stockfish's verdict on it (``best_move``,
``cp``, ``mate``, and every line in ``pvs``). No column is invented here; the mate convention
and the sign of a score are ``rukh.data.scoring``, shared with the DPO pairs.

Three labels:

``value``
    ``tanh(score / 400)`` from **White's** point of view, so the number is a position's value
    and not a player's, and a mate is ``±1`` rather than an outlier of ten thousand. Also
    published as five buckets (``< -200``, ``-200..-50``, ``-50..50``, ``50..200``, ``> 200``
    centipawns) so a regression and a classification can be compared on the same rows.

``blunder``
    whether ``last_move`` — the move that *led to* this position — threw away at least
    ``blunder_cp`` centipawns. It needs the previous position of the same game, and it is
    measured from the point of view of the side that moved: its best line there against what it
    actually got here, which is minus the value of this position for the opponent. A position
    whose predecessor is not in the table (the first ply, or a FEN that was deduplicated into
    another game) has no blunder label at all, ``null``, never ``0``.

``result``
    how the game the position came from ended: 0 White, 1 draw, 2 Black.

The split is by ``game_id`` and never by position. Two positions of the same game are not
independent: the second one is the first one plus a move, and a model that memorised the game
would score well on both. With a per-position split that leak is invisible in the metrics and
fatal in the demo, so ``test_labels`` proves no ``game_id`` is ever on both sides.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import polars as pl
from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.manifest import FileHash, Manifest
from rukh.data.scoring import MATE_SCORE
from rukh.data.uci import sha256_file
from rukh.paths import resolve

LABELS_FILE = "positions-labels.parquet"
RESULT_CLASSES: dict[str, int] = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
"""Result of the game the position came from: White wins, draw, Black wins."""
VALUE_EDGES: tuple[int, int, int, int] = (-200, -50, 50, 200)
"""Centipawn edges of the five value buckets, from White's point of view."""
SPLITS = ("train", "val")
SOURCE_COLUMNS = ["fen", "game_id", "ply", "last_move", "result", "phase", "cp", "mate"]


class LabelsConfig(BaseConfig):
    """Where the evaluations are, how a blunder is defined and how the split is drawn."""

    positions_eval: str = "data/evals/positions-eval.parquet"
    out_dir: str = "data/labels"
    value_scale: float = Field(default=400.0, gt=0.0)
    """``tanh(cp / value_scale)``: 400 centipawns (a rook) is about 0.76."""
    blunder_cp: int = Field(default=100, ge=1)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    seed: int = 42


def game_split(game_id: int, val_fraction: float, seed: int) -> str:
    """Which side of the split a game falls on: a pure function of its id and the seed.

    CRC-32 rather than ``hash()`` or a dataframe hash: it is the same number in every process,
    every polars version and every future run, which is what makes a split reproducible.
    """
    digest = zlib.crc32(f"{seed}:{game_id}".encode())
    return "val" if (digest % 1_000_000) / 1_000_000 < val_fraction else "train"


def row_order(game_id: int, ply: int, seed: int) -> int:
    """A stable per-row key: the order the label-count curve grows its nested subsets in."""
    return zlib.crc32(f"{seed}:{game_id}:{ply}".encode())


def _score_white() -> pl.Expr:
    """Centipawns from White's point of view, with a mate in ``n`` worth ``±(10000 - n)``."""
    mate = pl.col("mate")
    mated = pl.when(mate > 0).then(MATE_SCORE - mate).otherwise(-MATE_SCORE - mate)
    return pl.when(mate.is_not_null()).then(mated).otherwise(pl.col("cp")).cast(pl.Int64)


def _value_bucket(score: pl.Expr) -> pl.Expr:
    """Index of the five-bucket classification variant of ``value``."""
    bucket = pl.lit(0, dtype=pl.Int8)
    for index, edge in enumerate(VALUE_EDGES, start=1):
        bucket = pl.when(score >= edge).then(pl.lit(index, dtype=pl.Int8)).otherwise(bucket)
    return bucket


def build_labels(cfg: LabelsConfig, frame: pl.DataFrame | None = None) -> pl.DataFrame:
    """The labelled table; ``frame`` overrides reading ``cfg.positions_eval`` (used by tests)."""
    source = frame if frame is not None else pl.read_parquet(resolve(cfg.positions_eval))
    rows = (
        source.select(SOURCE_COLUMNS)
        .with_columns(
            score_white=_score_white(),
            mover=pl.when(pl.col("fen").str.split(" ").list.get(1) == "w").then(1).otherwise(-1),
        )
        .drop_nulls("score_white")
        .with_columns(score_mover=pl.col("mover") * pl.col("score_white"))
    )
    # The move into this position is judged against the best line of the position before it, and
    # from the mover's side: what it could have had there, minus what it actually got here
    # (which is minus the value of this position for the opponent, now to move).
    previous = rows.select(
        pl.col("game_id"),
        (pl.col("ply") + 1).alias("ply"),
        pl.col("score_mover").alias("before_best"),
    )
    joined = rows.join(previous, on=["game_id", "ply"], how="left").with_columns(
        loss_cp=(pl.col("before_best") + pl.col("score_mover")).cast(pl.Int64)
    )

    games = joined.select("game_id").unique().sort("game_id")
    split = pl.DataFrame(
        {
            "game_id": games["game_id"],
            "split": [game_split(int(g), cfg.val_fraction, cfg.seed) for g in games["game_id"]],
        }
    )
    labelled = (
        joined.join(split, on="game_id", how="left")
        .with_columns(
            value=(pl.col("score_white") / cfg.value_scale).tanh().cast(pl.Float32),
            value_bucket=_value_bucket(pl.col("score_white")),
            result_class=pl.col("result")
            .replace_strict(RESULT_CLASSES, default=None)
            .cast(pl.Int8),
            blunder=pl.when(pl.col("loss_cp").is_null())
            .then(None)
            .otherwise((pl.col("loss_cp") >= cfg.blunder_cp).cast(pl.Int8)),
        )
        .drop_nulls("result_class")
    )
    order = [
        row_order(int(game), int(ply), cfg.seed)
        for game, ply in zip(labelled["game_id"], labelled["ply"], strict=True)
    ]
    return labelled.with_columns(order=pl.Series("order", order, dtype=pl.UInt32)).select(
        "game_id",
        "ply",
        "fen",
        "last_move",
        "phase",
        "split",
        "value",
        "value_bucket",
        "blunder",
        "result_class",
        pl.col("score_white").alias("cp"),
        "loss_cp",
        "order",
    )


def label_subsets(frame: pl.DataFrame, fractions: list[float]) -> dict[float, pl.DataFrame]:
    """Nested subsets for the label-count curve: 10 % is inside 25 %, inside 50 %, inside 100 %.

    Nested on purpose. With independent samples, a dip at 50 % could just be an unlucky draw and
    the curve would measure sampling noise instead of the value of more labels; growing one set
    means every point is the previous one plus new rows.
    """
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError(f"every fraction must be in (0, 1], got {fractions}")
    ordered = frame.sort(["order", "game_id", "ply"])
    total = ordered.height
    return {
        fraction: ordered.head(max(1, round(total * fraction)))
        for fraction in sorted(set(fractions))
    }


def counts(frame: pl.DataFrame) -> dict[str, int]:
    """Row counts worth recording: total, per split and how many carry a blunder label."""
    per_split = {name: int(frame.filter(pl.col("split") == name).height) for name in SPLITS}
    return {
        "positions": int(frame.height),
        **per_split,
        "blunder_labelled": int(frame.drop_nulls("blunder").height),
        "blunders": int(frame.filter(pl.col("blunder") == 1).height),
    }


def run(cfg: LabelsConfig) -> Manifest:
    """Write ``out_dir/positions-labels.parquet`` plus the manifest, and return the manifest."""
    frame = build_labels(cfg)
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / LABELS_FILE
    frame.write_parquet(target, compression="zstd")
    manifest = Manifest(
        dataset="rukh-positions-eval",
        months=[],
        filters={
            "positions_eval": Path(cfg.positions_eval).as_posix(),
            "value_scale": cfg.value_scale,
            "value_edges": list(VALUE_EDGES),
            "blunder_cp": cfg.blunder_cp,
            "split": {"by": "game_id", "val_fraction": cfg.val_fraction, "seed": cfg.seed},
        },
        counts=counts(frame),
        files=[FileHash(path=LABELS_FILE, sha256=sha256_file(target), bytes=target.stat().st_size)],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
