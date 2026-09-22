"""Export the training curve of a run to ``artifacts/web/training-replay.json``.

Lab 3 of module M2. Reads the metrics MLflow stored during ``rukh train`` and writes **one
entry per checkpoint**: the slider of the ``TrainingReplay`` island walks the ``step-*.pt`` files
the run left behind, not the (much denser) MLflow logging steps, because every other number the
island can show — legality, Elo — only exists for a step whose weights are still on disk.

Only ``step`` is guaranteed in an entry. ``train_loss``, ``val_loss`` and ``val_top1`` are copied
from MLflow when that exact step was logged (with the shipped configs it always is: ``log_every``
and ``eval_every`` both divide ``ckpt_every``), ``legality`` needs ``--with-legality`` and ``elo``
needs ``--with-elo``.

``legality`` is the argmax rate of the design spec 02 (D-026): the single most likely token, no
temperature, no top-k and no mask. It is the bar ``GOAL.md`` sets at 99 %, and the one that means
something when it is plotted against the training step.

Usage (from the repository root):

    uv run python labs/m2/replay_export.py --run-name small-20260919-013000
    uv run python labs/m2/replay_export.py --run-id <mlflow run id> --with-legality
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from rukh import paths
from rukh.tracking import tracking_uri

log = logging.getLogger("replay_export")

SCHEMA = "rukh-training-replay/1"
CKPT_GLOB = "step-*.pt"
"""What ``rukh train`` writes every ``ckpt_every`` steps (plus the last one)."""

METRICS = {"train_loss": "train/loss", "val_loss": "val/loss", "val_top1": "val/top1"}
ELO_GAMES = 10
"""Games per rung when ``--with-elo`` is given: enough for a shape, far too few for a number."""


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


def run_checkpoints(directory: Path) -> dict[int, Path]:
    """``{step: path}`` of every ``step-*.pt`` in ``directory``, ordered by step."""
    found: dict[int, Path] = {}
    for path in Path(directory).glob(CKPT_GLOB):
        try:
            step = int(path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            log.debug("ignoring %s: not a step checkpoint", path)
            continue
        found[step] = path
    return dict(sorted(found.items()))


def replay_steps(metric_steps: Sequence[int], checkpoints: dict[int, Path]) -> list[int]:
    """The steps the island scrubs through: the logged steps that kept a checkpoint.

    The last logged step is always kept even when its checkpoint is gone (a run that was still
    training when this ran, or whose weights were pruned after publication): it is the model the
    rest of the table talks about, so leaving it out would be worse than leaving it thin.
    """
    chosen = {step for step in metric_steps if step in checkpoints}
    if metric_steps:
        chosen.add(max(metric_steps))
    return sorted(chosen)


def checkpoint_dir(run: mlflow.entities.Run, override: str | None) -> Path:
    """Where the run wrote its checkpoints: ``--checkpoints``, or ``out_dir/<run name>``."""
    if override:
        return Path(override)
    out_dir = str(run.data.params.get("out_dir", "checkpoints"))
    return paths.resolve(out_dir) / str(run.info.run_name)


def elo_of(model: Any, tok: Any, cfg: Any, games: int) -> float | None:
    """Estimated Elo from a handful of games per rung, or None when Stockfish is not around."""
    from rukh.engine import EngineNotFound
    from rukh.eval.elo import estimate, play_rungs

    try:
        records = play_rungs(
            model,
            tok,
            cfg.elo_rungs,
            games,
            cfg.sampling(),
            move_time=cfg.elo_move_time,
            max_plies=cfg.elo_max_plies,
        )
    except EngineNotFound as exc:
        log.warning("Elo skipped: %s", exc)
        return None
    if not records:
        return None
    return estimate(records, samples=cfg.bootstrap, seed=cfg.seed).elo


def evaluate_checkpoints(
    steps: Sequence[int],
    checkpoints: dict[int, Path],
    positions: int,
    with_elo: bool,
    elo_games: int,
) -> dict[int, dict[str, float]]:
    """Measure the legality (and optionally the Elo) of every checkpoint in ``steps``."""
    from rukh.eval.legality import legality, sample_positions
    from rukh.eval.suite import EvalConfig
    from rukh.tokenize.uci_vocab import UciTokenizer
    from rukh.train import load_model, pick_device

    cfg = EvalConfig(legality_positions=positions, elo_games=elo_games)
    games = paths.resolve(cfg.games)
    if not games.is_file():
        raise SystemExit(f"validation games not found at {games}: --with-legality needs the P1 cut")
    tok = UciTokenizer()
    prefixes = sample_positions(
        games, positions, tok, seed=cfg.seed, pool=cfg.position_pool, block=cfg.block
    )
    where = cfg.device or pick_device()
    measured: dict[int, dict[str, float]] = {}
    for step in steps:
        ckpt = checkpoints.get(step)
        if ckpt is None:
            log.warning("step %d has no checkpoint: it keeps its MLflow metrics only", step)
            continue
        model, _payload = load_model(ckpt, map_location=where)
        model = model.to(where).eval()
        # `mode="argmax"` on purpose: that is the definition the 99 % bar is written against.
        result = legality(model, tok, prefixes, cfg.sampling(), mode="argmax")
        values: dict[str, float] = {"legality": round(result.rate, 4)}
        line = f"  step {step:>6}  legality {result.rate:.3f}"
        if with_elo:
            elo = elo_of(model, tok, cfg, elo_games)
            if elo is not None:
                values["elo"] = round(elo, 1)
                line += f"  elo {elo:.0f}"
        measured[step] = values
        print(line)
    return measured


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name (default: latest run)")
    parser.add_argument("--run-id", default=None, help="MLflow run id, wins over --run-name")
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="directory holding the step-*.pt files (default: <out_dir>/<run name>)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON (default: artifacts/web/training-replay.json)",
    )
    parser.add_argument(
        "--with-legality",
        action="store_true",
        help="measure the unmasked argmax legality of every checkpoint (slow: one forward pass "
        "per sampled position per checkpoint)",
    )
    parser.add_argument(
        "--with-elo",
        action="store_true",
        help="also estimate the Elo of every checkpoint against Stockfish with --elo-games games "
        "per rung. VERY SLOW (eight rungs of real games per checkpoint) and, with so few games, "
        "only good for the shape of the curve: the published number comes from `rukh eval`. "
        "Implies --with-legality.",
    )
    parser.add_argument(
        "--elo-games",
        type=int,
        default=ELO_GAMES,
        help=f"games per Stockfish rung when --with-elo is given (default: {ELO_GAMES})",
    )
    parser.add_argument("--positions", type=int, default=500, help="positions per legality check")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri(create=False))
    run = find_run(args.run_name, args.run_id)
    run_id = run.info.run_id

    logged = {name: history(client, run_id, key) for name, key in METRICS.items()}
    metric_steps = sorted(set().union(*(series.keys() for series in logged.values())))
    if not metric_steps:
        raise SystemExit(f"run {run_id} has no train/val metrics")

    ckpt_dir = checkpoint_dir(run, args.checkpoints)
    checkpoints = run_checkpoints(ckpt_dir)
    steps = replay_steps(metric_steps, checkpoints)
    if not checkpoints:
        raise SystemExit(
            f"no {CKPT_GLOB} under {ckpt_dir}: the replay is one entry per checkpoint, so point "
            "--checkpoints at the directory the run wrote"
        )

    measured: dict[int, dict[str, float]] = {}
    if args.with_legality or args.with_elo:
        measured = evaluate_checkpoints(
            steps, checkpoints, args.positions, args.with_elo, args.elo_games
        )

    entries: list[dict[str, float | int]] = []
    for step in steps:
        entry: dict[str, float | int] = {"step": int(step)}
        for name, series in logged.items():
            if step in series:
                entry[name] = round(series[step], 4)
        entry.update(measured.get(step, {}))
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
            "checkpoints": ckpt_dir.as_posix(),
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{len(entries)} checkpoints -> {out}")


if __name__ == "__main__":
    main()
