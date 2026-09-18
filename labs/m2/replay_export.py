"""Export the training curve of a run to ``artifacts/web/training-replay.json``.

Lab 3 of module M2. Reads the metrics MLflow stored during ``rukh train`` and writes one
entry per logged step so the ``TrainingReplay`` island in the course can scrub through the
run. Optionally enriches the entries with the legality and Elo of each checkpoint, which is
what makes the "watch the game emerge" slider interesting; that part costs one quick
evaluation per checkpoint, so it is opt-in.

Usage (from the repository root):

    uv run python labs/m2/replay_export.py --run-name small-20260919-013000
    uv run python labs/m2/replay_export.py --run-id <mlflow run id> --with-legality
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from rukh import paths
from rukh.tracking import tracking_uri

SCHEMA = "rukh-training-replay/1"


def find_run(run_name: str | None, run_id: str | None) -> mlflow.entities.Run:
    """Return the requested run, or the most recent one when nothing is given."""
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri(create=False))
    if run_id:
        return client.get_run(run_id)
    experiment = client.get_experiment_by_name("rukh")
    if experiment is None:
        raise SystemExit("no 'rukh' experiment: run `rukh train` first")
    filter_string = f"attributes.run_name = '{run_name}'" if run_name else ""
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"no run matching {run_name or '<latest>'}")
    return runs[0]


def history(client: mlflow.tracking.MlflowClient, run_id: str, key: str) -> dict[int, float]:
    """Metric history as ``{step: value}`` (MLflow returns one record per logged point)."""
    try:
        return {m.step: m.value for m in client.get_metric_history(run_id, key)}
    except mlflow.exceptions.MlflowException:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None, help="MLflow run name (default: latest run)")
    parser.add_argument("--run-id", default=None, help="MLflow run id, wins over --run-name")
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON (default: artifacts/web/training-replay.json)",
    )
    parser.add_argument(
        "--with-legality",
        action="store_true",
        help="evaluate every checkpoint of the run (slow: one quick suite per checkpoint)",
    )
    parser.add_argument("--positions", type=int, default=500, help="positions per legality check")
    args = parser.parse_args()

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri(create=False))
    run = find_run(args.run_name, args.run_id)
    run_id = run.info.run_id

    train_loss = history(client, run_id, "train/loss")
    val_loss = history(client, run_id, "val/loss")
    val_top1 = history(client, run_id, "val/top1")
    steps = sorted(set(train_loss) | set(val_loss) | set(val_top1))
    if not steps:
        raise SystemExit(f"run {run_id} has no train/val metrics")

    extra: dict[int, dict[str, float]] = {}
    if args.with_legality:
        from rukh.eval.legality import legality
        from rukh.eval.suite import EvalConfig
        from rukh.train.checkpoint import load_model

        ckpt_dir = Path(str(run.data.params.get("out_dir", "checkpoints"))) / str(run.info.run_name)
        cfg = EvalConfig(legality_positions=args.positions)
        for ckpt in sorted(ckpt_dir.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1])):
            step = int(ckpt.stem.split("-")[1])
            model = load_model(ckpt, device=cfg.device)
            result = legality(model, cfg, mode="argmax")
            extra[step] = {"legality": result.rate}
            print(f"  step {step:>6}  legality {result.rate:.3f}")

    entries = []
    for step in steps:
        entry: dict[str, float | int] = {"step": int(step)}
        if step in train_loss:
            entry["train_loss"] = round(train_loss[step], 4)
        if step in val_loss:
            entry["val_loss"] = round(val_loss[step], 4)
        if step in val_top1:
            entry["val_top1"] = round(val_top1[step], 4)
        entry.update(extra.get(step, {}))
        entries.append(entry)

    default_out = paths.root() / "artifacts" / "web" / "training-replay.json"
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "steps": entries,
        "meta": {
            "run": run.info.run_name,
            "run_id": run_id,
            "preset": run.data.params.get("preset"),
            "max_steps": run.data.params.get("max_steps"),
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{len(entries)} steps -> {out}")


if __name__ == "__main__":
    main()
