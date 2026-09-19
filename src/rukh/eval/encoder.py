"""The encoder's own suite: blunder detection against the baseline, value correlation, result.

``rukh eval`` measures a decoder by the moves it plays; an encoder plays nothing, so it is
measured by what it knows about a position it has never seen. The held-out split is the one
``rukh.data.labels`` draws **by game** (two positions of the same game are not independent), and
every number here is computed on the very same rows for the model and for
``rukh.eval.heuristic``, because ``GOAL.md`` asks for a margin between the two and a margin
measured on two different sets is not a margin.

Four questions, four numbers:

``blunder``
    precision, recall and F1 over the rows that have a blunder label at all, for the encoder
    *and* for the material baseline. The headline is the difference in F1 points.

``value``
    Pearson **and** Spearman between the predicted value and Stockfish's ``cp``. Pearson asks
    whether the numbers line up, Spearman only whether the order does, and the two disagree
    exactly when the model has the ranking right and the scale wrong — which is what a bounded
    ``tanh`` head does to a score in centipawns, so both are reported. Spearman is Pearson over
    average ranks, written out here rather than imported: ``scipy`` is not a dependency.

``result``
    plain accuracy over the three classes.

``label curve``
    the 10/25/50/100 % points of the fine-tuning run, when the checkpoint carries them.

The heuristic is the slow part (a ``python-chess`` board per row), so its verdicts go through
``rukh.eval.cache`` under a key of their own: they do not depend on the weights, so a second
checkpoint evaluated on the same rows pays nothing for them.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from rukh import paths
from rukh.config import BaseConfig
from rukh.data.labels import LabelsConfig, build_labels
from rukh.eval import heuristic
from rukh.eval.cache import EvalCache, config_sha, file_sha
from rukh.eval.report import (
    ReportPaths,
    encoder_row_of,
    write_report_files,
)

log = logging.getLogger(__name__)

SUITE = "encoder"
HEURISTIC_SUITE = "heuristic"
HEURISTIC_KEY = "material-mobility-v1"
"""Cache key of the baseline: bump it when the heuristic's verdicts change."""
CURVE_KEY = "label_curve"
GOAL_MARGIN = 5.0
"""F1 points the encoder has to add to the baseline (``GOAL.md``)."""
DEFAULT_CONFIG = "encoder.yaml"


class EncoderEvalConfig(BaseConfig):
    """What the encoder suite measures and where it reads and writes."""

    stage: str | None = None
    """Name of the row in the results table; defaults to the checkpoint's run directory."""
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    split: str = "val"
    """Which side of the by-game split to measure; never ``train`` for a published number."""
    positions: int = 10_000
    """Cap on the rows evaluated; ``0`` means the whole split."""
    batch_size: int = 256
    threshold: float = 0.5
    """Probability above which the encoder's ``blunder`` logit counts as a blunder."""
    mobility_weight: float = heuristic.MOBILITY_WEIGHT
    blunder_material: float = heuristic.BLUNDER_MATERIAL
    seed: int = 42
    out_dir: str = "artifacts/eval"
    web_results: str | None = "artifacts/web/results.json"
    cache_db: str = "artifacts/eval/cache.sqlite"
    device: str | None = None
    """Where the model runs; ``None`` means ``rukh.train.pick_device()`` (CUDA when present)."""
    track: bool = True

    def cache_fields(self) -> dict[str, Any]:
        """The settings that change what a cached item means."""
        return {
            "split": self.split,
            "positions": self.positions,
            "threshold": self.threshold,
            "mobility_weight": self.mobility_weight,
            "blunder_material": self.blunder_material,
            "seed": self.seed,
            "labels": self.labels.model_dump(mode="json"),
        }

    def heuristic_fields(self) -> dict[str, Any]:
        """The settings a cached *baseline* verdict depends on: never the weights."""
        return {
            "heuristic": HEURISTIC_KEY,
            "mobility_weight": self.mobility_weight,
            "blunder_material": self.blunder_material,
        }


class ClassificationResult(BaseModel):
    """Precision, recall and F1 of one binary detector over one set of items."""

    model_config = ConfigDict(extra="forbid")

    name: str
    items: int
    positives: int
    """Rows whose label is 1."""
    predicted: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float


class CorrelationResult(BaseModel):
    """How well a predicted value tracks Stockfish's score."""

    model_config = ConfigDict(extra="forbid")

    name: str
    items: int
    pearson: float | None = None
    spearman: float | None = None


class CurvePoint(BaseModel):
    """One point of the label-count curve, as the fine-tuning run recorded it."""

    model_config = ConfigDict(extra="forbid")

    fraction: float
    train_labels: int | None = None
    metrics: dict[str, float] = {}


class EncoderResult(BaseModel):
    """Everything one encoder evaluation produced; serialised verbatim as ``results.json``."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    suite: str = SUITE
    checkpoint: str
    model_sha: str
    params: int
    date: str
    device: str = "cpu"
    run_id: str | None = None
    items: int = 0
    blunder_items: int = 0
    """Rows that carry a blunder label; the encoder and the baseline are scored on these."""
    encoder_blunder: ClassificationResult | None = None
    heuristic_blunder: ClassificationResult | None = None
    f1_margin: float | None = None
    """Encoder F1 minus baseline F1, in points; ``GOAL.md`` asks for at least five."""
    meets_goal: bool | None = None
    encoder_value: CorrelationResult | None = None
    heuristic_value: CorrelationResult | None = None
    result_accuracy: float | None = None
    label_curve: list[CurvePoint] = []
    notes: list[str] = []
    config: dict[str, Any] = {}


def config_path() -> Path:
    """Where the encoder suite's YAML lives inside the source tree."""
    return paths.package_root() / "configs" / "eval" / DEFAULT_CONFIG


def load_encoder_suite(config: Path | None = None) -> EncoderEvalConfig:
    """Load the encoder suite config: an explicit path wins over the packaged one."""
    from rukh.config import load_yaml

    return load_yaml(config if config is not None else config_path(), EncoderEvalConfig)


def classification(
    truth: Sequence[int], predicted: Sequence[int], name: str = "blunder"
) -> ClassificationResult:
    """Precision, recall and F1 of ``predicted`` against ``truth`` (both 0/1 per row)."""
    if len(truth) != len(predicted):
        raise ValueError(f"{len(truth)} labels against {len(predicted)} predictions")
    labels = np.asarray(truth, dtype=np.int64)
    calls = np.asarray(predicted, dtype=np.int64)
    hits = int(np.sum((labels == 1) & (calls == 1)))
    false_positives = int(np.sum((labels == 0) & (calls == 1)))
    false_negatives = int(np.sum((labels == 1) & (calls == 0)))
    precision = hits / (hits + false_positives) if hits + false_positives else 0.0
    recall = hits / (hits + false_negatives) if hits + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    correct = int(np.sum(labels == calls))
    return ClassificationResult(
        name=name,
        items=len(labels),
        positives=int(np.sum(labels == 1)),
        predicted=int(np.sum(calls == 1)),
        true_positives=hits,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=correct / len(labels) if len(labels) else 0.0,
    )


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson's correlation, or ``None`` when either side is constant (it is undefined)."""
    if len(x) != len(y):
        raise ValueError(f"{len(x)} values against {len(y)}")
    if len(x) < 2:
        return None
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()
    denominator = math.sqrt(float(first @ first) * float(second @ second))
    if denominator == 0.0:
        return None
    return float(first @ second / denominator)


def ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, one per value: tied values share the mean of the ranks they occupy."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        shared = (start + stop) / 2.0 + 1.0
        for position in range(start, stop + 1):
            out[order[position]] = shared
        start = stop + 1
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rank correlation: Pearson over average ranks, written out by hand."""
    if len(x) != len(y):
        raise ValueError(f"{len(x)} values against {len(y)}")
    if len(x) < 2:
        return None
    return pearson(ranks(list(x)), ranks(list(y)))


def correlation(
    predicted: Sequence[float], target: Sequence[float], name: str
) -> CorrelationResult:
    """Both correlations of one predictor against ``cp``, in one record."""
    return CorrelationResult(
        name=name,
        items=len(predicted),
        pearson=pearson(predicted, target),
        spearman=spearman(predicted, target),
    )


def build_items(cfg: EncoderEvalConfig, source: pl.DataFrame | None = None) -> pl.DataFrame:
    """The held-out rows to evaluate, each with the position that preceded it.

    The blunder label judges ``last_move``, the move that *led to* ``fen``, so the baseline needs
    the position before it: the same ``(game_id, ply - 1)`` self-join ``rukh.data.labels`` uses
    to build the label in the first place. Exactly the rows with a predecessor carry a label.
    """
    frame = build_labels(cfg.labels, source)
    previous = frame.select(
        pl.col("game_id"),
        (pl.col("ply") + 1).alias("ply"),
        pl.col("fen").alias("fen_before"),
    )
    items = (
        frame.join(previous, on=["game_id", "ply"], how="left")
        .filter(pl.col("split") == cfg.split)
        .sort(["order", "game_id", "ply"])
    )
    return items.head(cfg.positions) if cfg.positions else items


def predict(
    model: Any, fens: Sequence[str], batch_size: int = 256, device: str | None = None
) -> dict[str, np.ndarray]:
    """Run the three heads over every FEN: ``value``, ``blunder`` (probability) and ``result``."""
    import torch

    from rukh.models.squares import fen_to_tokens

    where = torch.device(device or "cpu")
    values: list[np.ndarray] = []
    blunders: list[np.ndarray] = []
    results: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(fens), max(1, batch_size)):
            chunk = list(fens[start : start + max(1, batch_size)])
            idx = torch.tensor([fen_to_tokens(fen) for fen in chunk], dtype=torch.long)
            outputs = model(idx.to(where))
            values.append(outputs["value"].detach().float().cpu().numpy())
            blunders.append(torch.sigmoid(outputs["blunder"].detach().float()).cpu().numpy())
            results.append(outputs["result"].detach().float().argmax(dim=-1).cpu().numpy())
    empty = np.zeros(0, dtype=np.float32)
    return {
        "value": np.concatenate(values) if values else empty,
        "blunder": np.concatenate(blunders) if blunders else empty,
        "result": np.concatenate(results) if results else np.zeros(0, dtype=np.int64),
    }


def _verdict(fen_before: str, move: str, cfg: EncoderEvalConfig) -> tuple[int, float]:
    verdict = heuristic.judge(fen_before, move, cfg.blunder_material, cfg.mobility_weight)
    return int(verdict.blunder), verdict.loss


def heuristic_predictions(
    items: pl.DataFrame, cfg: EncoderEvalConfig, cache: EvalCache | None = None
) -> dict[str, np.ndarray]:
    """The baseline's value for every row and its blunder call wherever a move can be judged.

    A row whose predecessor is missing gets ``nan`` for the blunder call, exactly like its label:
    the two sides of the comparison skip the same rows.
    """
    values = np.full(items.height, np.nan, dtype=np.float64)
    calls = np.full(items.height, np.nan, dtype=np.float64)
    columns = items.select("game_id", "ply", "fen", "fen_before", "last_move").rows()
    for index, (game_id, ply, fen, fen_before, last_move) in enumerate(columns):
        item_id = f"{game_id}:{ply}"
        payload = cache.get(HEURISTIC_SUITE, item_id) if cache is not None else None
        if payload is None:
            board_value = heuristic.value(chess.Board(fen), cfg.mobility_weight)
            call: float | None = None
            if fen_before and last_move:
                call = float(_verdict(str(fen_before), str(last_move), cfg)[0])
            payload = {"value": board_value, "blunder": call}
            if cache is not None:
                cache.put(HEURISTIC_SUITE, item_id, payload)
        values[index] = float(payload["value"])
        called = payload.get("blunder")
        calls[index] = np.nan if called is None else float(called)
    return {"value": values, "blunder": calls}


def label_curve_points(payload: dict[str, Any]) -> list[CurvePoint]:
    """The label-count curve a fine-tuning run may have written into its checkpoint.

    ``rukh train heads --curve`` trains one run per fraction; when the run that produced this
    checkpoint recorded them, they travel in the payload (or in its ``cfg``) under
    ``label_curve`` and are reported as they are. Nothing is invented when they are absent.
    """
    found = payload.get(CURVE_KEY)
    if found is None:
        found = (payload.get("cfg") or {}).get(CURVE_KEY)
    if not isinstance(found, list):
        return []
    points: list[CurvePoint] = []
    for entry in found:
        if not isinstance(entry, dict) or "fraction" not in entry:
            continue
        metrics = entry.get("metrics") or {}
        points.append(
            CurvePoint(
                fraction=float(entry["fraction"]),
                train_labels=(
                    int(entry["train_labels"]) if entry.get("train_labels") is not None else None
                ),
                metrics={
                    str(key): float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                },
            )
        )
    return sorted(points, key=lambda point: point.fraction)


def _finite(*arrays: np.ndarray) -> np.ndarray:
    """Mask of the rows where every array is a real number."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def measure(
    items: pl.DataFrame,
    predictions: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    cfg: EncoderEvalConfig,
    result: EncoderResult,
) -> EncoderResult:
    """Fill ``result`` with every metric the two predictors produced on the same rows."""
    cp = items["cp"].to_numpy().astype(np.float64)
    labels = items["blunder"].cast(pl.Float64).fill_null(np.nan).to_numpy().astype(np.float64)
    result.items = items.height
    scored = _finite(labels, baseline["blunder"])
    result.blunder_items = int(scored.sum())
    if result.blunder_items:
        truth = labels[scored].astype(np.int64)
        result.encoder_blunder = classification(
            truth, (predictions["blunder"][scored] >= cfg.threshold).astype(np.int64), "encoder"
        )
        result.heuristic_blunder = classification(
            truth, baseline["blunder"][scored].astype(np.int64), "heuristic"
        )
        margin = 100.0 * (result.encoder_blunder.f1 - result.heuristic_blunder.f1)
        result.f1_margin = margin
        result.meets_goal = margin >= GOAL_MARGIN
    if items.height:
        result.encoder_value = correlation(predictions["value"], cp, "encoder")
        result.heuristic_value = correlation(baseline["value"], cp, "heuristic")
        truth_result = items["result_class"].to_numpy().astype(np.int64)
        result.result_accuracy = float(np.mean(predictions["result"] == truth_result))
    return result


def evaluate_encoder(
    ckpt: Path | str,
    cfg: EncoderEvalConfig,
    use_cache: bool = True,
    device: str | None = None,
    source: pl.DataFrame | None = None,
) -> EncoderResult:
    """Measure one fine-tuned encoder; a missing label table becomes a note, never a crash."""
    from rukh.train import load_heads, pick_device

    path = Path(ckpt)
    where = device or cfg.device or pick_device()
    model, payload = load_heads(path, map_location=where)
    model = model.to(where).eval()
    log.info("evaluating %s on %s", path, where)
    notes: list[str] = [
        "the blunder F1 of the encoder and of the material baseline are measured on the same "
        f"rows of the held-out '{cfg.split}' split, which is drawn by game_id, never by position"
    ]
    result = EncoderResult(
        stage=cfg.stage or path.parent.name,
        checkpoint=path.as_posix(),
        model_sha=file_sha(path),
        params=sum(parameter.numel() for parameter in model.parameters()),
        date=datetime.now(UTC).date().isoformat(),
        device=str(where),
        label_curve=label_curve_points(payload),
        config=cfg.model_dump(mode="json"),
    )
    if not result.label_curve:
        notes.append(
            "the checkpoint carries no label-count curve: run `rukh train heads --curve` and "
            "evaluate one of its checkpoints to fill that table"
        )
    try:
        items = build_items(cfg, source)
    except FileNotFoundError as exc:
        notes.append(f"no labelled positions ({exc}): every metric was skipped")
        result.notes = notes
        return result
    if not items.height:
        notes.append(f"the '{cfg.split}' split is empty: every metric was skipped")
        result.notes = notes
        return result

    caches = _caches(cfg, result.model_sha, use_cache)
    try:
        predictions = predict(model, items["fen"].to_list(), cfg.batch_size, str(where))
        baseline = heuristic_predictions(items, cfg, caches[HEURISTIC_SUITE])
    finally:
        for cache in caches.values():
            cache.close()
    measure(items, predictions, baseline, cfg, result)
    if result.f1_margin is not None:
        notes.append(
            f"the encoder is {result.f1_margin:+.1f} F1 points from the baseline; GOAL.md asks "
            f"for at least {GOAL_MARGIN:+.0f}"
        )
    notes.append(
        "the baseline counts material (1/3/3/5/9) and mobility and looks one ply ahead at "
        "captures: it cannot see a positional sacrifice, and calls Fischer's 17...Be6 "
        "(Byrne-Fischer, 1956) a nine-point blunder"
    )
    result.notes = notes
    return result


def _caches(cfg: EncoderEvalConfig, model_sha: str, use_cache: bool) -> dict[str, EvalCache]:
    """One cache for the model's items and one for the baseline's, which outlive the weights."""
    database = paths.resolve(cfg.cache_db) if use_cache else None
    return {
        SUITE: EvalCache(
            database, model_sha, enabled=use_cache, config_sha=config_sha(cfg.cache_fields())
        ),
        HEURISTIC_SUITE: EvalCache(
            database,
            HEURISTIC_KEY,
            enabled=use_cache,
            config_sha=config_sha(cfg.heuristic_fields()),
        ),
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f} %"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _detector_rows(result: EncoderResult) -> list[str]:
    lines = [
        "| Detector | Items | Blunders | Flagged | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for measured in (result.encoder_blunder, result.heuristic_blunder):
        if measured is None:
            continue
        lines.append(
            f"| {measured.name} | {measured.items} | {measured.positives} | "
            f"{measured.predicted} | {_percent(measured.precision)} | "
            f"{_percent(measured.recall)} | {_percent(measured.f1)} |"
        )
    return lines


def render_markdown(result: EncoderResult) -> str:
    """The human-readable report: the margin first, then every breakdown."""
    encoder = result.encoder_blunder
    baseline = result.heuristic_blunder
    margin = "n/a" if result.f1_margin is None else f"{result.f1_margin:+.1f} points"
    lines: list[str] = [
        f"# Evaluation of `{result.stage}`",
        "",
        f"- Suite: `{result.suite}`",
        f"- Checkpoint: `{result.checkpoint}`",
        f"- Weights SHA-256: `{result.model_sha}`",
        f"- Parameters: {result.params:,}",
        f"- Device: `{result.device}`",
        f"- Date: {result.date}",
        f"- MLflow run: {result.run_id or 'not tracked'}",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Blunder F1, encoder | {_percent(encoder.f1 if encoder else None)} |",
        f"| Blunder F1, material baseline | {_percent(baseline.f1 if baseline else None)} |",
        f"| Margin over the baseline | {margin} |",
        f"| Value vs `cp`, Pearson | "
        f"{_number(result.encoder_value.pearson if result.encoder_value else None)} |",
        f"| Value vs `cp`, Spearman | "
        f"{_number(result.encoder_value.spearman if result.encoder_value else None)} |",
        f"| Result accuracy | {_percent(result.result_accuracy)} |",
        f"| Positions | {result.items:,} ({result.blunder_items:,} with a blunder label) |",
        "",
    ]
    if encoder is not None or baseline is not None:
        lines.extend(["## Blunder detection", "", *_detector_rows(result), ""])
    if result.encoder_value is not None:
        lines.extend(
            [
                "## Value against Stockfish",
                "",
                "Pearson asks whether the numbers line up, Spearman only whether the order does; "
                "a bounded `tanh` head over a score in centipawns can have the second right and "
                "the first wrong.",
                "",
                "| Predictor | Items | Pearson | Spearman |",
                "|---|---:|---:|---:|",
            ]
        )
        for measured in (result.encoder_value, result.heuristic_value):
            if measured is not None:
                lines.append(
                    f"| {measured.name} | {measured.items} | {_number(measured.pearson)} | "
                    f"{_number(measured.spearman)} |"
                )
        lines.append("")
    if result.label_curve:
        keys = sorted({key for point in result.label_curve for key in point.metrics})
        lines.extend(
            [
                "## Labels needed",
                "",
                "| Labels | Rows | " + " | ".join(f"`{key}`" for key in keys) + " |",
                "|---:|---:|" + "---:|" * len(keys),
            ]
        )
        for point in result.label_curve:
            cells = " | ".join(_number(point.metrics.get(key)) for key in keys)
            rows = "n/a" if point.train_labels is None else f"{point.train_labels:,}"
            lines.append(f"| {point.fraction:.0%} | {rows} | {cells} |")
        lines.append("")
    if result.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in result.notes)
        lines.append("")
    return "\n".join(lines)


def track_result(result: EncoderResult, cfg: EncoderEvalConfig) -> str | None:
    """Log the suite to the local MLflow store and return the run id."""
    import mlflow

    from rukh.tracking import start_run

    with start_run(
        f"eval-{result.stage}",
        {"suite": result.suite, "checkpoint": result.checkpoint, **cfg.model_dump(mode="json")},
        tags={"kind": "eval", "stage": result.stage, "model_sha": result.model_sha},
    ) as run:
        metrics = _metrics(result)
        if metrics:
            mlflow.log_metrics(metrics)
        return str(run.info.run_id)


def _metrics(result: EncoderResult) -> dict[str, float]:
    """Flat metrics for MLflow (only what was actually measured)."""
    metrics: dict[str, float] = {}
    for measured in (result.encoder_blunder, result.heuristic_blunder):
        if measured is not None:
            metrics[f"blunder_f1/{measured.name}"] = measured.f1
            metrics[f"blunder_precision/{measured.name}"] = measured.precision
            metrics[f"blunder_recall/{measured.name}"] = measured.recall
    for measured in (result.encoder_value, result.heuristic_value):
        if measured is not None and measured.pearson is not None:
            metrics[f"value_pearson/{measured.name}"] = measured.pearson
        if measured is not None and measured.spearman is not None:
            metrics[f"value_spearman/{measured.name}"] = measured.spearman
    if result.f1_margin is not None:
        metrics["blunder_f1_margin"] = result.f1_margin
    if result.result_accuracy is not None:
        metrics["result_accuracy"] = result.result_accuracy
    return metrics


def write_encoder_report(
    result: EncoderResult, out_dir: Path, web_results: Path | None = None
) -> ReportPaths:
    """Write ``report.md`` and ``results.json`` and upsert the encoder's row for the web."""
    return write_report_files(
        result.stage,
        render_markdown(result),
        json.loads(result.model_dump_json()),
        Path(out_dir),
        web_results,
        encoder_row_of(result),
    )


def run_encoder_suite(
    ckpt: Path | str,
    cfg: EncoderEvalConfig,
    use_cache: bool = True,
    device: str | None = None,
) -> tuple[EncoderResult, ReportPaths]:
    """Evaluate, track and report: the whole of ``rukh eval encoder`` in one call."""
    result = evaluate_encoder(ckpt, cfg, use_cache=use_cache, device=device)
    if cfg.track:
        result.run_id = track_result(result, cfg)
    report = write_encoder_report(
        result,
        paths.resolve(cfg.out_dir),
        paths.resolve(cfg.web_results) if cfg.web_results else None,
    )
    return result, report
