"""Tests for rukh.eval.encoder: F1 by hand, the two correlations, the report and the table."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import chess
import numpy as np
import polars as pl
import pytest
import torch
import yaml
from typer.testing import CliRunner

from helpers_labels import source_frame
from rukh.cli import app
from rukh.eval.encoder import (
    EncoderEvalConfig,
    EncoderResult,
    average_precision,
    best_threshold,
    blunder_notes,
    build_items,
    classification,
    evaluate_encoder,
    f1_at,
    label_curve_points,
    load_encoder_suite,
    measure_blunder,
    pearson,
    ranks,
    render_markdown,
    roc_auc,
    run_encoder_suite,
    spearman,
    threshold_halves,
)
from rukh.eval.report import RESULTS_NAME, EncoderWebRow, encoder_row_of, upsert_row
from rukh.models import EncoderConfig, PositionEncoder
from rukh.models.heads import MultiHead
from rukh.train import save_checkpoint

pytestmark = pytest.mark.unit

TOY = EncoderConfig(input="squares", n_layer=1, n_head=2, d_model=16, dropout=0.0)


def config(tmp_path: Path, **overrides: Any) -> EncoderEvalConfig:
    cfg = EncoderEvalConfig(
        split="val",
        positions=0,
        batch_size=4,
        out_dir=str(tmp_path / "eval"),
        web_results=str(tmp_path / "web" / "results.json"),
        cache_db=str(tmp_path / "eval" / "cache.sqlite"),
        device="cpu",
        track=False,
    )
    # Everything in the held-out split: these games exist to be measured, not to train on.
    cfg = cfg.model_copy(update={"labels": cfg.labels.model_copy(update={"val_fraction": 0.999})})
    return cfg.model_copy(update=overrides)


def checkpoint(tmp_path: Path, name: str = "encoder-heads", **payload: Any) -> Path:
    torch.manual_seed(0)
    model = MultiHead(PositionEncoder(TOY)).eval()
    path = tmp_path / name / "best.pt"
    save_checkpoint(
        path,
        step=10,
        model=model,
        optimizer=None,
        cfg={
            "pooling": "mean",
            "weights": {"value": 1.0, "blunder": 1.0, "result": 0.5},
            **payload,
        },
        model_cfg=TOY.model_dump(),
    )
    return path


# --- the metrics themselves ------------------------------------------------------------------


def test_f1_matches_the_number_computed_by_hand() -> None:
    # Six items: three blunders, of which the detector finds two, and it flags one quiet move.
    truth = [1, 1, 1, 0, 0, 0]
    predicted = [1, 1, 0, 1, 0, 0]
    result = classification(truth, predicted)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (2, 1, 1)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)  # 2 * (2/3 * 2/3) / (2/3 + 2/3)
    assert result.accuracy == pytest.approx(4 / 6)
    assert (result.items, result.positives, result.predicted) == (6, 3, 3)


def test_a_detector_that_flags_nothing_scores_zero_rather_than_dividing_by_zero() -> None:
    result = classification([1, 0, 1], [0, 0, 0])
    assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)


def test_mismatched_lengths_are_an_error() -> None:
    with pytest.raises(ValueError, match="2 labels against 3"):
        classification([1, 0], [1, 0, 1])


def test_a_perfect_predictor_correlates_one_and_a_shuffled_one_about_zero() -> None:
    import random

    truth = [float(value) for value in range(200)]
    assert pearson(truth, truth) == pytest.approx(1.0)
    assert spearman(truth, truth) == pytest.approx(1.0)
    assert pearson([-value for value in truth], truth) == pytest.approx(-1.0)
    shuffled = truth[:]
    random.Random(7).shuffle(shuffled)
    assert abs(pearson(shuffled, truth)) < 0.15
    assert abs(spearman(shuffled, truth)) < 0.15


def test_a_constant_predictor_has_no_correlation_at_all() -> None:
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert spearman([1.0], [2.0]) is None


def test_spearman_differs_from_pearson_on_a_monotone_but_nonlinear_case() -> None:
    """A bounded head against an unbounded score: the order is right, the scale is not.

    ``tanh(cp / 400)`` is exactly this shape, which is why both numbers are reported.
    """
    cp = [-2000.0, -800.0, -100.0, 0.0, 100.0, 800.0, 2000.0]
    predicted = [math.tanh(score / 400.0) for score in cp]
    assert spearman(predicted, cp) == pytest.approx(1.0)
    assert pearson(predicted, cp) < 0.95


def test_ranks_average_the_positions_of_a_tie() -> None:
    assert ranks([3.0, 1.0, 1.0, 2.0]) == [4.0, 1.5, 1.5, 3.0]


# --- the items ------------------------------------------------------------------------------


def test_every_item_carries_the_position_that_preceded_it(tmp_path: Path) -> None:
    items = build_items(config(tmp_path), source_frame())
    assert items.height == 16  # two games of eight plies
    first = items.filter((pl.col("game_id") == 1) & (pl.col("ply") == 1)).to_dicts()[0]
    assert first["fen_before"] is None and first["blunder"] is None
    blunder = items.filter((pl.col("game_id") == 1) & (pl.col("ply") == 5)).to_dicts()[0]
    assert blunder["blunder"] == 1  # 3.Qxf7+ threw the queen away
    assert chess.Board(blunder["fen_before"]).is_legal(chess.Move.from_uci(blunder["last_move"]))


# --- the suite ------------------------------------------------------------------------------


def test_the_suite_scores_both_detectors_on_the_same_rows(tmp_path: Path) -> None:
    result = evaluate_encoder(
        checkpoint(tmp_path), config(tmp_path), use_cache=False, source=source_frame()
    )
    assert result.items == 16
    assert result.blunder_items == 14  # the first ply of each game has no predecessor
    assert result.encoder_blunder is not None and result.heuristic_blunder is not None
    assert result.encoder_blunder.items == result.heuristic_blunder.items == result.blunder_items
    # The baseline sees 3.Qxf7+ and 3...Kxf7 is a capture it also judges; whatever it says, the
    # margin is the difference of the two F1 values in points.
    assert result.f1_margin == pytest.approx(
        100.0 * (result.encoder_blunder.f1 - result.heuristic_blunder.f1)
    )
    assert result.meets_goal is (result.f1_margin >= 5.0)
    assert result.encoder_value is not None and result.result_accuracy is not None
    assert result.heuristic_value is not None


def test_the_baseline_finds_the_queen_blunder_of_the_game(tmp_path: Path) -> None:
    result = evaluate_encoder(
        checkpoint(tmp_path), config(tmp_path), use_cache=False, source=source_frame()
    )
    assert result.heuristic_blunder is not None
    assert result.heuristic_blunder.true_positives >= 2  # 3.Qxf7+ in both games


def test_a_missing_label_table_becomes_a_note(tmp_path: Path, rukh_home: Path) -> None:
    cfg = config(tmp_path, labels=EncoderEvalConfig().labels)
    result = evaluate_encoder(checkpoint(tmp_path), cfg, use_cache=False)
    assert result.encoder_blunder is None
    assert any("no labelled positions" in note for note in result.notes)


def test_the_baseline_verdicts_are_cached_across_checkpoints(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    evaluate_encoder(checkpoint(tmp_path, "first"), cfg, source=source_frame())
    database = Path(cfg.cache_db)
    assert database.is_file()
    import sqlite3

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM items WHERE suite = 'heuristic'"
        ).fetchone()[0]
    # Eight, not sixteen: the fixture is the same game twice, and the verdicts are keyed by the
    # position and the move judged, so the second copy hits the cache instead of writing its own
    # row under a `game_id:ply` nobody can match against anything.
    assert rows == 8
    # A different checkpoint reuses them: the baseline does not depend on the weights.
    evaluate_encoder(checkpoint(tmp_path, "second"), cfg, source=source_frame())
    with sqlite3.connect(database) as connection:
        again = connection.execute(
            "SELECT COUNT(*) FROM items WHERE suite = 'heuristic'"
        ).fetchone()[0]
    assert again == rows


def test_the_label_curve_is_read_from_the_checkpoint_when_it_carries_one(tmp_path: Path) -> None:
    curve = [
        {"fraction": 0.5, "train_labels": 500, "metrics": {"val/loss": 0.9}},
        {"fraction": 0.1, "train_labels": 100, "metrics": {"val/loss": 1.2}},
    ]
    points = label_curve_points({"cfg": {"label_curve": curve}})
    assert [point.fraction for point in points] == [0.1, 0.5]
    assert points[0].metrics == {"val/loss": 1.2}
    assert label_curve_points({"cfg": {}}) == []
    path = checkpoint(tmp_path, label_curve=curve)
    result = evaluate_encoder(path, config(tmp_path), use_cache=False, source=source_frame())
    assert [point.fraction for point in result.label_curve] == [0.1, 0.5]
    assert "## Labels needed" in render_markdown(result)


# --- the report and the shared table ----------------------------------------------------------


def test_the_report_round_trips_and_names_the_baseline(tmp_path: Path, rukh_home: Path) -> None:
    from rukh.eval.encoder import EncoderResult

    cfg = config(tmp_path)
    frame = source_frame()
    source = rukh_home / "positions-eval.parquet"
    frame.write_parquet(source)
    cfg = cfg.model_copy(
        update={"labels": cfg.labels.model_copy(update={"positions_eval": str(source)})}
    )
    result, report = run_encoder_suite(checkpoint(tmp_path), cfg, use_cache=False, device="cpu")
    text = Path(report.markdown).read_text(encoding="utf-8")
    assert "# Evaluation of `encoder-heads`" in text
    assert "| Blunder F1, material baseline |" in text
    assert "Byrne-Fischer" in text
    reloaded = EncoderResult.model_validate_json(Path(report.results).read_text(encoding="utf-8"))
    assert Path(report.results).name == RESULTS_NAME
    assert reloaded.model_dump() == result.model_dump()


def test_the_encoder_row_does_not_disturb_the_decoder_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    decoder = {
        "stage": "small",
        "params": 39_000_000,
        "legality": 0.99,
        "elo": 1234.0,
        "date": "2026-09-19",
    }
    path.write_text(json.dumps({"rows": [decoder]}), encoding="utf-8")
    row = EncoderWebRow(stage="encoder", params=15_052_800, blunder_f1=0.4, date="2026-09-19")
    rows = upsert_row(path, row)
    assert len(rows) == 2
    kept = next(item for item in rows if item["stage"] == "small")
    assert kept == decoder  # not one key added, not one removed
    written = next(item for item in rows if item["stage"] == "encoder")
    assert written["kind"] == "encoder" and written["blunder_f1"] == 0.4
    assert "legality" not in written


def test_the_row_of_a_result_carries_both_f1_values(tmp_path: Path) -> None:
    result = evaluate_encoder(
        checkpoint(tmp_path), config(tmp_path), use_cache=False, source=source_frame()
    )
    row = encoder_row_of(result)
    assert row.stage == "encoder-heads" and row.kind == "encoder"
    assert row.blunder_f1 == result.encoder_blunder.f1
    assert row.blunder_f1_heuristic == result.heuristic_blunder.f1
    assert row.positions == result.items


# --- the command line -------------------------------------------------------------------------


def test_the_packaged_config_loads() -> None:
    cfg = load_encoder_suite()
    assert cfg.split == "val" and cfg.threshold == 0.5
    assert cfg.labels.blunder_cp == 100


def test_the_cli_evaluates_an_encoder_and_writes_the_table(tmp_path: Path, rukh_home: Path) -> None:
    source = rukh_home / "positions-eval.parquet"
    source_frame().write_parquet(source)
    cfg = config(tmp_path)
    payload = cfg.model_dump(mode="json")
    payload["labels"]["positions_eval"] = str(source)
    payload["labels"]["val_fraction"] = 0.999
    config_path = tmp_path / "encoder.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    invocation = CliRunner().invoke(
        app,
        [
            "eval",
            "encoder",
            "--model",
            str(checkpoint(tmp_path)),
            "--config",
            str(config_path),
            "--stage",
            "encoder-probe",
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    assert "blunder:" in invocation.output and "heuristic" in invocation.output
    table = json.loads((tmp_path / "web" / "results.json").read_text(encoding="utf-8"))
    assert [row["stage"] for row in table["rows"]] == ["encoder-probe"]
    assert table["rows"][0]["kind"] == "encoder"


def test_the_decoder_spelling_of_eval_still_needs_a_model() -> None:
    invocation = CliRunner().invoke(app, ["eval"])
    assert invocation.exit_code == 2
    assert "--model is required" in invocation.output


# --- the label-count curve, end to end ----------------------------------------------------------


def test_a_two_point_curve_reaches_the_evaluation_report(rukh_home: Path, tmp_path: Path) -> None:
    """`rukh train heads --curve` writes the points; `rukh eval encoder` renders them."""
    from rukh.train import HeadsConfig, label_curve

    source = rukh_home / "evals" / "positions-eval.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    source_frame().write_parquet(source)
    labels = {
        "positions_eval": "evals/positions-eval.parquet",
        "out_dir": "labels",
        # game 1 lands in train and game 2 in val with the default seed, so both sides have rows.
        "val_fraction": 0.5,
    }
    heads = HeadsConfig(
        model=TOY,
        labels=labels,
        batch_size=4,
        grad_accum=1,
        warmup=1,
        max_steps=2,
        precision="fp32",
        compile=False,
        eval_every=2,
        eval_batches=1,
        ckpt_every=2,
        log_every=1,
        run_name="curve-heads",
        unique_run_name=False,
    )
    results = label_curve(heads, [0.5, 1.0], device="cpu")
    assert [result.fraction for result in results] == [0.5, 1.0]

    cfg = config(tmp_path).model_copy(
        update={"labels": EncoderEvalConfig().labels.model_copy(update=labels)}
    )
    measured = evaluate_encoder(
        Path(results[-1].checkpoint), cfg, use_cache=False, source=source_frame()
    )
    assert [point.fraction for point in measured.label_curve] == [0.5, 1.0]
    assert measured.label_curve[0].train_labels == results[0].train_labels
    assert measured.label_curve[0].train_labels < measured.label_curve[1].train_labels
    assert all("no label-count curve" not in note for note in measured.notes)
    assert "## Labels needed" in render_markdown(measured)


def test_the_curve_key_is_the_same_string_on_both_sides() -> None:
    from rukh.eval.encoder import CURVE_KEY
    from rukh.train.checkpoint import CURVE_KEY as WRITTEN

    assert CURVE_KEY == WRITTEN  # the reader and the writer, one string


# --- the second acceptance criterion ------------------------------------------------------------


def test_the_value_criterion_is_checked_against_the_bounded_score(tmp_path: Path) -> None:
    from rukh.eval.encoder import GOAL_VALUE_CORRELATION, measure

    cfg = config(tmp_path)
    items = build_items(cfg, source_frame())
    rows = items.height
    perfect = np.tanh(items["cp"].to_numpy().astype(float) / cfg.labels.value_scale)
    result = EncoderResult(
        stage="toy", checkpoint="x", model_sha="sha", params=1, date="2026-09-19"
    )
    predictions = {
        "value": perfect,
        "blunder": np.zeros(rows),
        "result": np.zeros(rows, dtype=np.int64),
    }
    baseline = {"value": np.zeros(rows), "blunder": np.full(rows, np.nan)}
    measure(items, predictions, baseline, cfg, result)
    assert result.encoder_value is not None
    assert result.encoder_value.target == "tanh(cp / 400)"
    assert result.encoder_value.spearman == pytest.approx(1.0)
    assert result.value_correlation_meets_goal is True
    # A predictor with the ranking upside down fails the bar without touching the F1 one.
    result_bad = result.model_copy(deep=True)
    measure(items, {**predictions, "value": -perfect}, baseline, cfg, result_bad)
    assert result_bad.value_correlation_meets_goal is False
    assert GOAL_VALUE_CORRELATION == 0.80


# --- the operating point ------------------------------------------------------------------------


def synthetic(
    high: float, low: float, other_high: float, other_low: float, games: int = 40, rows: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels, probabilities and game ids where the two halves want different thresholds.

    Every game gets ``rows`` positions, one of which is a blunder. The probabilities of the
    ``tune`` half separate at one place and those of the ``score`` half at another, so a
    threshold copied from the first half is measurably wrong on the second: that is the only way
    to prove the harness is not quietly choosing the threshold on the rows it reports.
    """
    identifiers = np.repeat(np.arange(1, games + 1), rows)
    truth = np.tile(np.array([1] + [0] * (rows - 1)), games)
    tune = threshold_halves(identifiers, 42)
    probability = np.where(
        tune,
        np.where(truth == 1, high, low),
        np.where(truth == 1, other_high, other_low),
    )
    return truth, probability, identifiers


def scored(truth: np.ndarray, probability: np.ndarray, games: np.ndarray) -> EncoderResult:
    """Run the blunder half of the suite over arrays, with no model and no chess in the way."""
    result = EncoderResult(
        stage="toy", checkpoint="x", model_sha="sha", params=1, date="2026-09-19"
    )
    calls = np.zeros(len(truth), dtype=np.int64)
    return measure_blunder(truth, probability, calls, games, EncoderEvalConfig(), result)


def test_the_threshold_is_chosen_on_tune_and_applied_to_score() -> None:
    truth, probability, games = synthetic(high=0.30, low=0.10, other_high=0.60, other_low=0.45)
    result = scored(truth, probability, games)
    tune = threshold_halves(games, 42)

    assert result.threshold_split_degenerate is False
    assert result.tune_items + result.score_items == result.blunder_items == len(truth)
    assert set(games[tune].tolist()).isdisjoint(set(games[~tune].tolist()))
    assert result.tune_games + result.score_games == len(set(games.tolist()))

    # The threshold is the `tune` half's own optimum, and it is the whole of what `tune` is for.
    chosen = best_threshold(truth[tune], probability[tune])[0]
    assert result.threshold_tuned == pytest.approx(chosen)
    assert result.tune_f1 == pytest.approx(1.0)
    # It is then applied verbatim to the other half, which separates somewhere else: the number
    # that gets reported is well below what choosing the threshold on these rows would have given.
    assert result.encoder_blunder is not None
    assert result.encoder_blunder.items == result.score_items
    assert result.encoder_blunder.f1 < best_threshold(truth[~tune], probability[~tune])[1]
    flagged = (probability[~tune] >= result.threshold_tuned).astype(np.int64)
    assert result.encoder_blunder.f1 == pytest.approx(classification(truth[~tune], flagged).f1)


def test_a_head_whose_probabilities_never_reach_one_half_still_gets_a_number() -> None:
    """The real case: every probability below 0.5, so the fixed threshold fires on nothing."""
    truth, probability, games = synthetic(high=0.20, low=0.02, other_high=0.26, other_low=0.01)
    result = scored(truth, probability, games)
    assert result.encoder_blunder is not None and result.encoder_blunder_fixed is not None
    assert result.threshold_tuned is not None and result.threshold_tuned < 0.5
    assert result.encoder_blunder_fixed.f1 == 0.0  # nothing is flagged at 0.5 and F1 says nothing
    assert result.encoder_blunder.f1 == pytest.approx(1.0)
    assert result.encoder_blunder_ranking is not None
    assert result.encoder_blunder_ranking.roc_auc == pytest.approx(1.0)
    assert result.encoder_blunder_ranking.max_score == pytest.approx(0.26)
    assert result.blunder_base_rate == pytest.approx(0.1)


def test_a_degenerate_split_is_reported_rather_than_hidden() -> None:
    """Two games on the same side: there is no half to hold back, and the report says so."""
    truth, probability, games = synthetic(0.9, 0.1, 0.9, 0.1, games=2, rows=6)
    result = scored(truth, probability, games)
    assert result.threshold_split_degenerate is True
    assert result.tune_items == result.score_items == result.blunder_items
    assert any("upper bound" in note for note in blunder_notes(result))
    assert "Warning:" in render_markdown(result)


def test_a_separable_case_scores_one_everywhere() -> None:
    truth = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    scores = [0.90, 0.80, 0.70, 0.20, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01]
    threshold, f1 = best_threshold(truth, scores)
    assert f1 == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.7
    assert roc_auc(truth, scores) == pytest.approx(1.0)
    assert average_precision(truth, scores) == pytest.approx(1.0)


def test_a_random_predictor_scores_a_coin_flip_and_the_base_rate() -> None:
    rng = np.random.default_rng(11)
    truth = (rng.random(20_000) < 0.037).astype(np.int64)
    scores = rng.random(20_000)
    base_rate = float(truth.mean())
    assert roc_auc(truth, scores) == pytest.approx(0.5, abs=0.02)
    assert average_precision(truth, scores) == pytest.approx(base_rate, abs=0.01)
    # A constant predictor ranks nothing: the average ranks make it exactly a coin flip, and its
    # precision-recall curve is the flat line at the base rate. Neither is a sorting accident.
    flat = np.full(len(truth), 0.3)
    assert roc_auc(truth, flat) == pytest.approx(0.5)
    assert average_precision(truth, flat) == pytest.approx(base_rate)


def test_the_tuned_f1_is_never_below_the_fixed_one_on_the_rows_it_was_chosen_on() -> None:
    """The only guarantee a sweep can honestly give, which is why it is measured elsewhere."""
    rng = np.random.default_rng(5)
    for trial in range(20):
        truth = (rng.random(400) < 0.05).astype(np.int64)
        scores = np.clip(rng.random(400) * 0.3 + truth * 0.1, 0.0, 1.0)
        threshold, tuned = best_threshold(truth, scores, include=0.5)
        assert tuned >= f1_at(truth, scores, 0.5), trial
        assert tuned == pytest.approx(f1_at(truth, scores, threshold)), trial


def test_neither_ranking_metric_is_defined_without_both_classes() -> None:
    assert roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None
    assert roc_auc([0, 0, 0], [0.1, 0.2, 0.3]) is None
    assert average_precision([0, 0, 0], [0.1, 0.2, 0.3]) is None
    with pytest.raises(ValueError, match="2 labels against 3"):
        roc_auc([1, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="2 labels against 3"):
        average_precision([1, 0], [0.1, 0.2, 0.3])


def test_the_report_renders_both_operating_points_and_the_caveat() -> None:
    truth, probability, games = synthetic(high=0.20, low=0.02, other_high=0.26, other_low=0.01)
    result = scored(truth, probability, games)
    text = render_markdown(result)
    tuned = f"{result.threshold_tuned:.4g}"
    assert f"| Blunder F1, encoder (tuned, `p >= {tuned}`) |" in text
    assert "| Blunder F1, encoder (fixed, `p >= 0.5`) |" in text
    assert "| Blunder ROC AUC |" in text and "| Blunder average precision |" in text
    assert "accuracy is meaningless" in text
    assert 'always answers "no blunder" scores 90.0 %' in text
    assert f"chose the threshold `p >= {tuned}`" in text
    assert "rule, no threshold to tune" in text
    assert "no `game_id` is in both halves" in text
    assert f"{result.tune_items:,} rows, {result.tune_games:,} games" in text


def test_the_caveat_and_both_operating_points_reach_the_model_card() -> None:
    from rukh.publish.model import (
        ENCODER_CARD_TEMPLATE,
        ModelPublishConfig,
        encoder_card_context,
        render_card,
    )

    truth, probability, games = synthetic(high=0.20, low=0.02, other_high=0.26, other_low=0.01)
    result = scored(truth, probability, games)
    context = encoder_card_context(
        "chorcat/rukh-encoder",
        "encoder",
        ModelPublishConfig(),
        {"params": 15_052_800, "tokenizer": "squares"},
        json.loads(result.model_dump_json()),
        None,
        [],
    )
    card = render_card(context, ENCODER_CARD_TEMPLATE)
    assert "accuracy is meaningless" in card
    assert "| Blunder ROC AUC |" in card
    assert f"`p >= {result.threshold_tuned:.4g}`" in card
    assert "it has no threshold" in card


def test_the_web_row_carries_both_operating_points() -> None:
    truth, probability, games = synthetic(high=0.20, low=0.02, other_high=0.26, other_low=0.01)
    result = scored(truth, probability, games)
    row = encoder_row_of(result)
    assert row.blunder_f1 == result.encoder_blunder.f1
    assert row.blunder_f1_fixed == result.encoder_blunder_fixed.f1
    assert row.blunder_threshold == result.threshold_tuned
    assert row.blunder_roc_auc == pytest.approx(1.0)
    assert row.blunder_base_rate == pytest.approx(0.1)
