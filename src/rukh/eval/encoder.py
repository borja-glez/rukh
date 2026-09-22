"""The encoder's own suite: blunder detection against the baseline, value correlation, result.

``rukh eval`` measures a decoder by the moves it plays; an encoder plays nothing, so it is
measured by what it knows about a position it has never seen. The held-out split is the one
``rukh.data.labels`` draws **by game** (two positions of the same game are not independent), and
every number here is computed on the very same rows for the model and for
``rukh.eval.heuristic``, because ``docs/acceptance.md`` asks for a margin between the two and a
margin
measured on two different sets is not a margin.

Four questions, four numbers:

``blunder``
    precision, recall and F1 over the rows that have a blunder label at all, for the encoder
    *and* for the material baseline. The headline is the difference in F1 points.

    A blunder is rare (about 3.7 % of the labelled rows) and that changes what an F1 at a
    fixed threshold means. The head's probabilities are not calibrated (nobody calibrated
    them), so at 0.5 a perfectly informative head can fire on nothing at all and score an F1 of
    zero while ranking the positions almost right. Scoring the *threshold* instead of the
    representation is not a measurement, so the rows are cut in two **by game** (the policy of
    ``rukh.data.labels``, for the same reason): the ``tune`` half chooses the threshold that
    maximises F1, and the ``score`` half is where the reported F1, precision and recall are
    measured. The number at the fixed 0.5 is printed next to it, on the same ``score`` half, so
    nothing is hidden, and ``ROC AUC`` and ``average precision`` are printed too because they
    are the two summaries that do not depend on an operating point at all. The baseline is a
    hard yes/no rule with no threshold to tune, which the report says out loud: the comparison
    gives the model a sweep the baseline cannot have.

``value``
    Spearman **and** Pearson between the predicted value and ``tanh(cp / 400)``, the bounded
    score the head was trained on — never raw ``cp``, where a forced mate is ``±9 99x`` and a
    handful of rows would decide Pearson for the whole set. Spearman is the headline (it asks
    only whether the ranking is right, which is what "the model knows which position is better"
    means) and the ``docs/acceptance.md`` bar of 0.80 is checked against it; Pearson is printed next
    to it
    because the two disagree exactly when the order is right and the scale is not. Spearman is
    Pearson over average ranks, written out here rather than imported: ``scipy`` is not a
    dependency.

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
import zlib
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
from rukh.eval.cache import weights_sha as tensor_sha
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
"""The payload key ``rukh.train.checkpoint.CURVE_KEY`` writes; a test pins the two together."""
GOAL_MARGIN = 5.0
"""F1 points the encoder has to add to the baseline (``docs/acceptance.md``)."""
TUNE_HALF = "tune"
"""Half of the labelled rows the operating point is chosen on, and never scored on."""
SCORE_HALF = "score"
"""Half of the labelled rows every reported blunder number is measured on."""
GOAL_VALUE_CORRELATION = 0.80
"""The second acceptance criterion of ``docs/acceptance.md``: value against Stockfish, at least 0.8.

Measured as **Spearman against the bounded score**, and both halves of that sentence matter.
The target is ``tanh(score / 400)``, so the head cannot reproduce centipawns and was never
asked to: correlating a bounded output against raw ``cp`` would compare a number in ``(-1, 1)``
with one that a forced mate sends to ``±9 99x``, and a handful of mates would decide Pearson for
the whole set. Rank correlation is also the honest reading of "the model knows which position is
better", which is what the bar is about. Pearson on the same bounded pair is reported next to it.
"""
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

    def heuristic_fields(self) -> dict[str, Any]:
        """The settings a cached *baseline* verdict depends on: never the weights.

        ``labels`` is in here even though the verdict is a function of the position alone,
        because it says *which table the positions came from*: pointing ``positions_eval`` at a
        different parquet has to invalidate the cache, and a cached verdict is keyed by the
        position itself (see ``heuristic_predictions``) precisely so that two tables sharing a
        position share the answer instead of overwriting each other's.
        """
        return {
            "heuristic": HEURISTIC_KEY,
            "mobility_weight": self.mobility_weight,
            "blunder_material": self.blunder_material,
            "labels": self.labels.model_dump(mode="json"),
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


class RankingResult(BaseModel):
    """What a detector's *score* is worth before anyone picks a threshold for it.

    ROC AUC is the probability that a random blunder is ranked above a random quiet move, and
    average precision is the area under the precision-recall curve, which is the right summary
    when the positive class is rare: a coin flip scores 0.5 on the first and the base rate on
    the second, so the two say different things about the same ranking. The span of the
    probabilities is here as well, because it is what makes a fixed threshold reasonable or
    absurd: a head whose largest output is 0.26 fires on nothing at 0.5.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    items: int
    positives: int
    base_rate: float
    """Share of the rows that are blunders; also what average precision is compared against."""
    roc_auc: float | None = None
    average_precision: float | None = None
    min_score: float | None = None
    max_score: float | None = None
    mean_score: float | None = None


class CorrelationResult(BaseModel):
    """How well a predicted value tracks Stockfish's score."""

    model_config = ConfigDict(extra="forbid")

    name: str
    items: int
    pearson: float | None = None
    spearman: float | None = None
    target: str = "tanh(cp / value_scale)"
    """What the prediction was correlated against; never raw ``cp``.

    See ``GOAL_VALUE_CORRELATION`` for why."""


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
    weights_sha: str | None = None
    """SHA-256 of the tensors alone, so a pulled copy of the same weights is recognised."""
    params: int
    date: str
    device: str = "cpu"
    run_id: str | None = None
    items: int = 0
    blunder_items: int = 0
    """Rows that carry a blunder label; they are split in two halves by ``game_id``."""
    blunder_positives: int = 0
    """How many of those rows are blunders."""
    blunder_base_rate: float | None = None
    """Share of blunders among the labelled rows: the number every metric here has to be read
    against, and the reason accuracy says nothing (see ``BASE_RATE_CAVEAT``)."""
    tune_items: int = 0
    """Rows of the ``tune`` half: where the threshold is chosen and where nothing is reported."""
    score_items: int = 0
    """Rows of the ``score`` half: where every reported blunder number is measured."""
    tune_games: int = 0
    score_games: int = 0
    """Games behind each half; no ``game_id`` is ever in both."""
    threshold_fixed: float | None = None
    """The configured threshold, reported as a second operating point and never as the headline."""
    threshold_tuned: float | None = None
    """The threshold that maximises F1 on the ``tune`` half; the headline is measured with it."""
    tune_f1: float | None = None
    """F1 of the tuned threshold **on the half it was chosen on**: the optimistic number, kept
    next to the honest one so the distance between the two is visible."""
    threshold_split_degenerate: bool = False
    """True when the by-game split could not give both halves blunders, so the threshold had to
    be chosen and measured on the same rows; the report says so and the number is optimistic."""
    encoder_blunder: ClassificationResult | None = None
    """The headline: the encoder at the tuned threshold, on the ``score`` half."""
    encoder_blunder_fixed: ClassificationResult | None = None
    """The same rows at the fixed threshold, so the operating point cannot hide anything."""
    encoder_blunder_ranking: RankingResult | None = None
    """ROC AUC and average precision on the ``score`` half: no threshold involved."""
    heuristic_blunder: ClassificationResult | None = None
    """The baseline on the very same ``score`` half. It is a yes/no rule: nothing was tuned."""
    f1_margin: float | None = None
    """Encoder F1 minus baseline F1, in points; ``docs/acceptance.md`` asks for at least five."""
    meets_goal: bool | None = None
    """The **blunder** criterion alone; ``docs/acceptance.md`` has two and this is the first."""
    encoder_value: CorrelationResult | None = None
    heuristic_value: CorrelationResult | None = None
    value_correlation_meets_goal: bool | None = None
    """The second criterion: Spearman of the value head against the bounded score >= 0.80."""
    meets_all_goals: bool | None = None
    """Both criteria at once, which is what "P3 is done" means."""
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


def base_rate_caveat(base_rate: float | None) -> str:
    """The one sentence that has to travel with every number in this section.

    It is in the report and in the model card because it is the transferable lesson of the
    module: on a rare class, the two numbers everybody reaches for first say nothing.
    """
    share = "the blunder base rate" if base_rate is None else f"{base_rate * 100:.1f} %"
    always = "" if base_rate is None else f" scores {(1 - base_rate) * 100:.1f} %"
    return (
        f"a blunder is rare ({share} of the labelled rows), so **accuracy is meaningless** here: "
        f'a model that always answers "no blunder"{always} without knowing anything about '
        "chess. F1 at an arbitrary threshold is nearly as bad, because an uncalibrated sigmoid "
        "can rank the positions well and still put every probability below 0.5: that number "
        "measures the operating point, not the representation. This is why the threshold is "
        f"chosen on a `{TUNE_HALF}` half and the F1 is reported on a `{SCORE_HALF}` half, and "
        "why ROC AUC and average precision, which no threshold can flatter, are reported next "
        "to it"
    )


def sentence(text: str) -> str:
    """The same clause as a sentence: the notes start in lower case, the prose does not.

    ``str.capitalize`` is not it: it would lower-case ``ROC AUC`` and ``F1`` on the way past.
    """
    text = text.strip()
    return text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")


def threshold_half(game_id: int, seed: int) -> str:
    """Which half of the labelled rows a game falls on: ``tune`` or ``score``.

    By game and never by position, exactly like ``rukh.data.labels.game_split`` and for exactly
    the same reason: two positions of the same game are one move apart, so choosing a threshold
    on one and scoring it on the other would tune on the rows being scored through the back
    door. CRC-32 of a salted string, so the halves are the same in every process and every run,
    and the salt keeps this split independent of the train/val one.
    """
    return TUNE_HALF if zlib.crc32(f"{seed}:threshold:{game_id}".encode()) % 2 == 0 else SCORE_HALF


def threshold_halves(games: Sequence[int], seed: int) -> np.ndarray:
    """Boolean mask of the rows that belong to the ``tune`` half, one entry per row."""
    return np.array([threshold_half(int(game), seed) == TUNE_HALF for game in games], dtype=bool)


def roc_auc(truth: Sequence[int], scores: Sequence[float]) -> float | None:
    """Area under the ROC curve, as the rank sum of the positives (Mann-Whitney U).

    Written out rather than imported: ``scipy`` and ``scikit-learn`` are not dependencies. Ties
    are handled by the average ranks of ``ranks``, which is what gives a constant predictor
    exactly 0.5 instead of 0 or 1 depending on how the sort happened to break the tie.
    """
    labels = np.asarray(truth, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(values):
        raise ValueError(f"{len(labels)} labels against {len(values)} scores")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        return None
    rank = np.asarray(ranks(values.tolist()), dtype=np.float64)
    rank_sum = float(rank[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(truth: Sequence[int], scores: Sequence[float]) -> float | None:
    """Area under the precision-recall curve, summed over the steps of the recall.

    ``sum (R_n - R_{n-1}) * P_n`` over the distinct scores, taken from the highest down. Rows
    that share a score are one step, because a threshold cannot separate them; without that,
    a predictor that outputs the same number everywhere would score 1.0 by sorting luck. A
    random ranking scores the base rate, which is why this is the summary that means something
    when the positive class is rare.
    """
    labels = np.asarray(truth, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(values):
        raise ValueError(f"{len(labels)} labels against {len(values)} scores")
    positives = int(np.sum(labels == 1))
    if not positives:
        return None
    order = np.argsort(-values, kind="stable")
    ordered = values[order]
    hits = np.cumsum(labels[order] == 1)
    seen = np.arange(1, len(labels) + 1, dtype=np.float64)
    last = np.append(ordered[1:] != ordered[:-1], True)  # the end of each group of equal scores
    recall = hits[last] / positives
    precision = hits[last] / seen[last]
    steps = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(steps * precision))


def ranking(truth: Sequence[int], scores: Sequence[float], name: str = "encoder") -> RankingResult:
    """Everything about a detector's ranking that no threshold can change."""
    labels = np.asarray(truth, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    return RankingResult(
        name=name,
        items=len(labels),
        positives=positives,
        base_rate=positives / len(labels) if len(labels) else 0.0,
        roc_auc=roc_auc(labels, values),
        average_precision=average_precision(labels, values),
        min_score=float(values.min()) if len(values) else None,
        max_score=float(values.max()) if len(values) else None,
        mean_score=float(values.mean()) if len(values) else None,
    )


def f1_at(truth: Sequence[int], scores: Sequence[float], threshold: float) -> float:
    """F1 of ``scores >= threshold`` against ``truth``."""
    calls = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)
    return classification(truth, calls).f1


def best_threshold(
    truth: Sequence[int], scores: Sequence[float], include: float = 0.5
) -> tuple[float, float]:
    """The threshold that maximises F1 on these rows, and the F1 it reaches.

    The candidates are the distinct scores (every threshold that can change a single call) plus
    ``include``, the configured one, so the answer is never *worse* than the fixed threshold on
    the rows it was chosen on — which is the only guarantee a sweep can honestly give, and the
    reason the reported number is measured somewhere else. Ties go to the lowest threshold: on a
    rare class, the lower one flags more and is the less lucky of the two.
    """
    labels = np.asarray(truth, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(values):
        raise ValueError(f"{len(labels)} labels against {len(values)} scores")
    positives = int(np.sum(labels == 1))
    if not len(labels) or not positives:
        return float(include), f1_at(labels, values, include)
    order = np.argsort(-values, kind="stable")
    ordered = values[order]
    hits = np.cumsum(labels[order] == 1)
    seen = np.arange(1, len(labels) + 1, dtype=np.float64)
    last = np.append(ordered[1:] != ordered[:-1], True)
    recall = hits[last] / positives
    precision = hits[last] / seen[last]
    total = precision + recall
    f1 = np.where(total > 0, 2 * precision * recall / np.where(total > 0, total, 1.0), 0.0)
    candidates = [
        (float(value), float(score)) for value, score in zip(ordered[last], f1, strict=True)
    ]
    candidates.append((float(include), f1_at(labels, values, include)))
    candidates.sort(key=lambda pair: (-pair[1], pair[0]))
    return candidates[0]


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
    predicted: Sequence[float],
    target: Sequence[float],
    name: str,
    target_name: str = "tanh(cp / value_scale)",
) -> CorrelationResult:
    """Both correlations of one predictor against the bounded score, in one record."""
    return CorrelationResult(
        name=name,
        items=len(predicted),
        pearson=pearson(predicted, target),
        spearman=spearman(predicted, target),
        target=target_name,
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


def encoder_items(items: pl.DataFrame, model: Any, labels: LabelsConfig) -> list[list[int]]:
    """Tokenize the rows in the scheme the model was trained on, whichever that is.

    ``squares`` reads the FEN and nothing else; ``moves`` needs the line that reached the
    position, so ``rukh.data.labels.game_moves`` joins the P1 games back in exactly as
    ``rukh.train.heads`` does — the evaluation has to feed the model the shape it was fine-tuned
    on, and hard-coding ``fen_to_tokens`` here would silently evaluate a ``moves`` encoder on
    tokens from another vocabulary.
    """
    from rukh.data.labels import game_moves
    from rukh.train.heads import LabelledPositions

    scheme = str(model.encoder.cfg.input)
    moves = game_moves(items, labels.games_dir) if scheme == "moves" else None
    dataset = LabelledPositions(items, scheme, moves, model.encoder.cfg.block)
    return [dataset.tokens(index) for index in range(len(dataset))]


def predict(
    model: Any,
    items: Sequence[Sequence[int]],
    batch_size: int = 256,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Run the three heads over the tokenized rows: ``value``, ``blunder`` (a probability) and
    ``result``; short sequences are padded and the padding is masked out."""
    import torch

    from rukh.train.heads import collate

    where = torch.device(device or "cpu")
    values: list[np.ndarray] = []
    blunders: list[np.ndarray] = []
    results: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(items), max(1, batch_size)):
            chunk = [list(tokens) for tokens in items[start : start + max(1, batch_size)]]
            batch = collate(
                [{"idx": torch.tensor(tokens, dtype=torch.long)} for tokens in chunk]  # type: ignore[misc]
            )
            outputs = model(batch["idx"].to(where), batch["attention_mask"].to(where))
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
    for index, (_game_id, _ply, fen, fen_before, last_move) in enumerate(columns):
        # Keyed by the position judged, never by ``game_id:ply``: the payload is a function of
        # these three strings and of nothing else, so the same position reached from another
        # table reuses the answer, and a *different* position that happens to sit at the same
        # ply of the same game cannot silently inherit a stale verdict. ``fen_before`` is part
        # of the key because ``fen`` plus ``last_move`` does not recover the captured piece,
        # and the baseline's whole judgement is the material difference between the two.
        item_id = f"{fen}|{fen_before or ''}|{last_move or ''}"
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


def measure_blunder(
    truth: np.ndarray,
    probability: np.ndarray,
    baseline_calls: np.ndarray,
    games: Sequence[int],
    cfg: EncoderEvalConfig,
    result: EncoderResult,
) -> EncoderResult:
    """Score the blunder detector at an operating point that was not chosen on these rows.

    The labelled rows are cut in two **by game**: ``tune`` chooses the threshold that maximises
    F1, ``score`` is where the F1, the precision and the recall that go into the report and into
    the ``docs/acceptance.md`` comparison are measured. The baseline is measured on the same
    ``score`` rows;
    it has no threshold to tune, so the comparison hands the model a sweep the rule cannot have,
    and the report says so rather than leaving the reader to notice.

    When the split cannot give both halves a blunder — a handful of games, or every blunder in
    one of them — there is no honest way to hold rows back: the threshold is then chosen and
    measured on the same rows, ``threshold_split_degenerate`` is set and the report prints the
    warning next to the number.
    """
    result.blunder_items = len(truth)
    result.blunder_positives = int(np.sum(truth == 1))
    result.blunder_base_rate = float(np.mean(truth == 1)) if len(truth) else None
    result.threshold_fixed = float(cfg.threshold)
    tune = threshold_halves(games, cfg.seed)
    score = ~tune
    identifiers = np.asarray(games)
    usable = bool(
        truth[tune].sum()
        and truth[score].sum()
        and (truth[tune] == 0).any()
        and (truth[score] == 0).any()
    )
    if not usable:
        result.threshold_split_degenerate = True
        tune = np.ones(len(truth), dtype=bool)
        score = np.ones(len(truth), dtype=bool)
    result.tune_items = int(tune.sum())
    result.score_items = int(score.sum())
    result.tune_games = int(len(np.unique(identifiers[tune])))
    result.score_games = int(len(np.unique(identifiers[score])))
    threshold, tune_f1 = best_threshold(truth[tune], probability[tune], cfg.threshold)
    result.threshold_tuned = threshold
    result.tune_f1 = tune_f1
    result.encoder_blunder = classification(
        truth[score], (probability[score] >= threshold).astype(np.int64), "encoder"
    )
    result.encoder_blunder_fixed = classification(
        truth[score], (probability[score] >= cfg.threshold).astype(np.int64), "encoder"
    )
    result.encoder_blunder_ranking = ranking(truth[score], probability[score], "encoder")
    result.heuristic_blunder = classification(truth[score], baseline_calls[score], "heuristic")
    margin = 100.0 * (result.encoder_blunder.f1 - result.heuristic_blunder.f1)
    result.f1_margin = margin
    result.meets_goal = margin >= GOAL_MARGIN
    return result


def measure(
    items: pl.DataFrame,
    predictions: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    cfg: EncoderEvalConfig,
    result: EncoderResult,
) -> EncoderResult:
    """Fill ``result`` with every metric the two predictors produced on the same rows."""
    # The value head is trained on ``tanh(cp / value_scale)`` and is measured against it. Raw
    # ``cp`` would put a mate at +-9 99x against an output that cannot leave (-1, 1), and those
    # few rows would set the Pearson of the whole set on their own.
    bounded = np.tanh(items["cp"].to_numpy().astype(np.float64) / float(cfg.labels.value_scale))
    labels = items["blunder"].cast(pl.Float64).fill_null(np.nan).to_numpy().astype(np.float64)
    result.items = items.height
    scored = _finite(labels, baseline["blunder"])
    result.blunder_items = int(scored.sum())
    if result.blunder_items:
        measure_blunder(
            labels[scored].astype(np.int64),
            predictions["blunder"][scored].astype(np.float64),
            baseline["blunder"][scored].astype(np.int64),
            items["game_id"].to_numpy()[scored],
            cfg,
            result,
        )
    if items.height:
        target = f"tanh(cp / {cfg.labels.value_scale:g})"
        result.encoder_value = correlation(predictions["value"], bounded, "encoder", target)
        result.heuristic_value = correlation(baseline["value"], bounded, "heuristic", target)
        headline = result.encoder_value.spearman
        result.value_correlation_meets_goal = (
            None if headline is None else headline >= GOAL_VALUE_CORRELATION
        )
        truth_result = items["result_class"].to_numpy().astype(np.int64)
        result.result_accuracy = float(np.mean(predictions["result"] == truth_result))
    if result.meets_goal is not None and result.value_correlation_meets_goal is not None:
        result.meets_all_goals = result.meets_goal and result.value_correlation_meets_goal
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
        f"rows of the held-out '{cfg.split}' split, which is drawn by game_id, never by "
        f"position, and those rows are cut in two by game_id again: the '{TUNE_HALF}' half "
        f"chooses the threshold and the '{SCORE_HALF}' half is what gets reported"
    ]
    result = EncoderResult(
        stage=cfg.stage or path.parent.name,
        checkpoint=path.as_posix(),
        model_sha=file_sha(path),
        weights_sha=tensor_sha(path),
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

    cache = _heuristic_cache(cfg, use_cache)
    try:
        tokenized = encoder_items(items, model, cfg.labels)
        predictions = predict(model, tokenized, cfg.batch_size, str(where))
        baseline = heuristic_predictions(items, cfg, cache)
    finally:
        cache.close()
    measure(items, predictions, baseline, cfg, result)
    if result.f1_margin is not None:
        notes.append(
            f"the encoder is {result.f1_margin:+.1f} F1 points from the baseline; "
            "docs/acceptance.md asks "
            f"for at least {GOAL_MARGIN:+.0f}"
        )
    if result.encoder_value is not None:
        notes.append(
            f"the value head is correlated against `{result.encoder_value.target}`, the bounded "
            "score it is trained on, and not against raw `cp`, where a forced mate is worth "
            "±9 99x and a few rows would decide Pearson for the whole set; the "
            "docs/acceptance.md bar of "
            f"{GOAL_VALUE_CORRELATION:.2f} is read on Spearman"
        )
    notes.append(
        "the baseline counts material (1/3/3/5/9) and mobility and looks one ply ahead at "
        "captures: it cannot see a positional sacrifice, and calls Fischer's 17...Be6 "
        "(Byrne-Fischer, 1956) a nine-point blunder"
    )
    notes.append(
        "the two blunder detectors do not see the same thing: the baseline is given the "
        "predecessor position and the move that was played, while the encoder is given only the "
        "resulting position and has to infer that something was thrown away. That is the "
        "comparison docs/acceptance.md asks for, but it is not a level playing field"
    )
    notes.extend(blunder_notes(result))
    result.notes = notes
    return result


def blunder_notes(result: EncoderResult) -> list[str]:
    """How the blunder numbers have to be read; the report and the model card share this text."""
    fixed = result.encoder_blunder_fixed
    if result.encoder_blunder is None or fixed is None:
        return []
    if result.threshold_tuned is None or result.threshold_fixed is None:
        return []
    notes = [
        f"the blunder threshold {result.threshold_tuned:.4g} was chosen on the "
        f"'{TUNE_HALF}' half ({result.tune_items:,} rows, {result.tune_games:,} games) by "
        f"maximising F1 there, and the reported numbers are measured on the '{SCORE_HALF}' half "
        f"({result.score_items:,} rows, {result.score_games:,} games)"
        + ("" if result.threshold_split_degenerate else ", which no threshold ever saw")
        + f"; the same rows at the fixed threshold {result.threshold_fixed:.4g} give an F1 of "
        f"{fixed.f1:.4f} against {result.encoder_blunder.f1:.4f} tuned",
        "the material baseline is a hard yes/no rule: it has no threshold, so nothing was tuned "
        "on its side and it got no half to tune on. The margin therefore compares a model at its "
        "best operating point against a rule at its only one",
        base_rate_caveat(result.blunder_base_rate),
    ]
    if result.threshold_split_degenerate:
        notes.append(
            f"the by-game split could not give both halves blunders ({result.blunder_positives} "
            f"of {result.blunder_items} labelled rows are blunders), so the threshold was chosen "
            "and measured on the same rows: this F1 is an upper bound, not a held-out number"
        )
    return notes


def _heuristic_cache(cfg: EncoderEvalConfig, use_cache: bool) -> EvalCache:
    """The one cache this suite has: the baseline's verdicts, which outlive the weights.

    There is deliberately no cache for the model's own predictions. A forward pass over ten
    thousand positions is seconds, the heuristic is a ``python-chess`` board per row, and a
    second cache keyed by the weights would only ever be read when the very same checkpoint is
    evaluated twice on the very same rows.
    """
    database = paths.resolve(cfg.cache_db) if use_cache else None
    return EvalCache(
        database,
        HEURISTIC_KEY,
        enabled=use_cache,
        config_sha=config_sha(cfg.heuristic_fields()),
    )


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f} %"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _verdict_cell(met: bool | None) -> str:
    """How an acceptance criterion reads in the report: measured, missed, or not measured."""
    return "not measured" if met is None else ("yes" if met else "no")


def _detector_row(measured: ClassificationResult | None, operating_point: str) -> str | None:
    if measured is None:
        return None
    return (
        f"| {measured.name} | {operating_point} | {measured.items} | {measured.positives} | "
        f"{measured.predicted} | {_percent(measured.precision)} | "
        f"{_percent(measured.recall)} | {_percent(measured.f1)} |"
    )


def _blunder_section(result: EncoderResult) -> list[str]:
    """The whole blunder block: how the operating point was chosen, then every number it gave."""
    tuned = "n/a" if result.threshold_tuned is None else f"{result.threshold_tuned:.4g}"
    fixed = "n/a" if result.threshold_fixed is None else f"{result.threshold_fixed:.4g}"
    lines = [
        "## Blunder detection",
        "",
        sentence(base_rate_caveat(result.blunder_base_rate)),
        "",
        f"The {result.blunder_items:,} labelled rows are cut in two by `game_id`, never by "
        "position, the policy the held-out split itself uses, because two positions of the same "
        f"game are one move apart. The `{TUNE_HALF}` half ({result.tune_items:,} rows, "
        f"{result.tune_games:,} games) chose the threshold `p >= {tuned}` by maximising F1 "
        f"there; the `{SCORE_HALF}` half ({result.score_items:,} rows, {result.score_games:,} "
        "games) is where every number below is measured"
        + ("." if result.threshold_split_degenerate else ", and no `game_id` is in both halves."),
        "",
        "The material baseline is a hard yes/no rule: it has no threshold, so it was given no "
        "half to tune on and nothing was swept on its side. The margin below compares the model "
        "at its best operating point against the rule at its only one, and that is a courtesy "
        "the model receives, not one the baseline does.",
        "",
        "| Detector | Operating point | Items | Blunders | Flagged | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [
        _detector_row(result.encoder_blunder, f"tuned, `p >= {tuned}` (chosen on `{TUNE_HALF}`)"),
        _detector_row(result.encoder_blunder_fixed, f"fixed, `p >= {fixed}`"),
        _detector_row(result.heuristic_blunder, "rule, no threshold to tune"),
    ]
    lines.extend(row for row in rows if row is not None)
    lines.append("")
    if result.tune_f1 is not None:
        lines.extend(
            [
                f"The same threshold reaches an F1 of {_percent(result.tune_f1)} on the "
                f"`{TUNE_HALF}` half it was chosen on. The distance between that and the "
                f"`{SCORE_HALF}` half above is what picking an operating point costs, and it is "
                "the reason the two halves are not the same rows.",
                "",
            ]
        )
    if result.threshold_split_degenerate:
        lines.extend(
            [
                "**Warning:** the by-game split could not give both halves blunders, so the "
                "threshold was chosen and measured on the same rows. The F1 above is an upper "
                "bound, not a held-out number.",
                "",
            ]
        )
    measured = result.encoder_blunder_ranking
    if measured is not None:
        span = "n/a"
        if measured.min_score is not None and measured.max_score is not None:
            span = f"{measured.min_score:.4f} to {measured.max_score:.4f}"
            if measured.mean_score is not None:
                span += f" (mean {measured.mean_score:.4f})"
        lines.extend(
            [
                "### Without an operating point",
                "",
                "ROC AUC is the probability that a random blunder is ranked above a random quiet "
                "move (0.5 is a coin flip); average precision is the area under the "
                "precision-recall curve, and a random ranking scores the base rate, which is "
                "printed next to it. Neither depends on a threshold, so neither can be flattered "
                "by choosing one. The span of the probabilities is here because it is what makes "
                "a fixed threshold reasonable or absurd.",
                "",
                "| Detector | Items | Blunders | Base rate | ROC AUC | Average precision | "
                "Probability span |",
                "|---|---:|---:|---:|---:|---:|---|",
                f"| {measured.name} | {measured.items} | {measured.positives} | "
                f"{_percent(measured.base_rate)} | {_number(measured.roc_auc)} | "
                f"{_number(measured.average_precision)} | {span} |",
                "",
            ]
        )
    return lines


def render_markdown(result: EncoderResult) -> str:
    """The human-readable report: the margin first, then every breakdown."""
    encoder = result.encoder_blunder
    fixed = result.encoder_blunder_fixed
    baseline = result.heuristic_blunder
    measured = result.encoder_blunder_ranking
    margin = "n/a" if result.f1_margin is None else f"{result.f1_margin:+.1f} points"
    target = result.encoder_value.target if result.encoder_value else "tanh(cp / value_scale)"
    tuned = "n/a" if result.threshold_tuned is None else f"{result.threshold_tuned:.4g}"
    at_fixed = "n/a" if result.threshold_fixed is None else f"{result.threshold_fixed:.4g}"
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
        f"| Blunder F1, encoder (tuned, `p >= {tuned}`) | "
        f"{_percent(encoder.f1 if encoder else None)} |",
        f"| Blunder F1, encoder (fixed, `p >= {at_fixed}`) | "
        f"{_percent(fixed.f1 if fixed else None)} |",
        f"| Blunder F1, material baseline | {_percent(baseline.f1 if baseline else None)} |",
        f"| Margin over the baseline | {margin} |",
        f"| Margin >= {GOAL_MARGIN:.0f} points | {_verdict_cell(result.meets_goal)} |",
        f"| Blunder ROC AUC | {_number(measured.roc_auc if measured else None)} |",
        f"| Blunder average precision | "
        f"{_number(measured.average_precision if measured else None)} |",
        f"| Blunder base rate | {_percent(result.blunder_base_rate)} |",
        f"| Value vs `{target}`, Spearman (headline) | "
        f"{_number(result.encoder_value.spearman if result.encoder_value else None)} |",
        f"| Value vs `{target}`, Pearson | "
        f"{_number(result.encoder_value.pearson if result.encoder_value else None)} |",
        f"| Value correlation >= {GOAL_VALUE_CORRELATION:.2f} | "
        f"{_verdict_cell(result.value_correlation_meets_goal)} |",
        f"| Result accuracy | {_percent(result.result_accuracy)} |",
        f"| Positions | {result.items:,} ({result.blunder_items:,} with a blunder label, "
        f"{result.score_items:,} of them scored) |",
        "",
    ]
    if encoder is not None or baseline is not None:
        lines.extend(_blunder_section(result))
    if result.encoder_value is not None:
        lines.extend(
            [
                "## Value against Stockfish",
                "",
                f"Both predictors are correlated against `{target}`, the bounded score the value "
                "head is trained on, never against raw `cp`: a forced mate is worth ±9 99x "
                "there and would decide Pearson for the whole set on its own. Spearman asks "
                "only whether the ranking is right and is the number the goal is checked "
                "against; Pearson also asks whether the scale is.",
                "",
                "| Predictor | Items | Target | Spearman | Pearson |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for measured in (result.encoder_value, result.heuristic_value):
            if measured is not None:
                lines.append(
                    f"| {measured.name} | {measured.items} | `{measured.target}` | "
                    f"{_number(measured.spearman)} | {_number(measured.pearson)} |"
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
    for measured, name in (
        (result.encoder_blunder, "encoder"),
        (result.encoder_blunder_fixed, "encoder_fixed"),
        (result.heuristic_blunder, "heuristic"),
    ):
        if measured is not None:
            metrics[f"blunder_f1/{name}"] = measured.f1
            metrics[f"blunder_precision/{name}"] = measured.precision
            metrics[f"blunder_recall/{name}"] = measured.recall
    if result.encoder_blunder_ranking is not None:
        auc = result.encoder_blunder_ranking.roc_auc
        precision = result.encoder_blunder_ranking.average_precision
        if auc is not None:
            metrics["blunder_roc_auc"] = auc
        if precision is not None:
            metrics["blunder_average_precision"] = precision
    for name, value in (
        ("blunder_threshold", result.threshold_tuned),
        ("blunder_base_rate", result.blunder_base_rate),
        ("blunder_f1_tune", result.tune_f1),
    ):
        if value is not None:
            metrics[name] = float(value)
    for measured in (result.encoder_value, result.heuristic_value):
        if measured is not None and measured.pearson is not None:
            metrics[f"value_pearson/{measured.name}"] = measured.pearson
        if measured is not None and measured.spearman is not None:
            metrics[f"value_spearman/{measured.name}"] = measured.spearman
    if result.f1_margin is not None:
        metrics["blunder_f1_margin"] = result.f1_margin
    for name, met in (
        ("meets_goal_blunder", result.meets_goal),
        ("meets_goal_value", result.value_correlation_meets_goal),
        ("meets_goal_all", result.meets_all_goals),
    ):
        if met is not None:
            metrics[name] = float(met)
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
