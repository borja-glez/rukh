"""The evaluation suite: one command that produces one row of the results table.

``run_suite`` loads a checkpoint, measures legality, next-move accuracy, puzzles and Elo, and
hands the result to ``report``. Every part is optional at run time: a missing validation
parquet, a missing puzzle parquet or a missing Stockfish binary becomes a note in the report
instead of a crash, because a partial evaluation that says what it could not measure is more
useful than no evaluation at all.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rukh import paths
from rukh.config import BaseConfig
from rukh.eval.accuracy import AccuracyResult, accuracy
from rukh.eval.cache import EvalCache, file_sha
from rukh.eval.elo import DEFAULT_RUNGS, EloResult, EloRung, estimate, play_rungs
from rukh.eval.legality import LegalityResult, legality, sample_positions
from rukh.eval.puzzles import PuzzleResult, load_puzzles, model_source, run_puzzles
from rukh.eval.report import ReportPaths, write_report
from rukh.infer import SampleConfig
from rukh.tokenize.uci_vocab import UciTokenizer

SUITES = ("full", "quick")


class EvalConfig(BaseConfig):
    """What a suite measures and where it reads and writes."""

    stage: str | None = None
    """Name of the row in the results table; defaults to the checkpoint's run directory."""
    games: str = "data/uci/year=2025/month=02/games.parquet"
    puzzles: str = "data/puzzles/puzzles.parquet"
    puzzle_split: str = "test"
    legality_positions: int = 10_000
    accuracy_positions: int = 10_000
    position_pool: int = 50_000
    puzzles_per_band: int = 2_000
    elo_games: int = 100
    elo_rungs: list[EloRung] = Field(default_factory=lambda: list(DEFAULT_RUNGS))
    elo_move_time: float = 0.05
    elo_max_plies: int | None = None
    bootstrap: int = 1_000
    temperature: float = 0.6
    top_k: int | None = 20
    block: int = 200
    seed: int = 42
    out_dir: str = "artifacts/eval"
    web_results: str | None = "artifacts/web/results.json"
    cache_db: str = "artifacts/eval/cache.sqlite"
    track: bool = True
    """Log the suite to the local MLflow store (the model card reads those metrics back)."""

    def sampling(self) -> SampleConfig:
        """The sampler the Elo games use: masked, seeded, as the demo plays."""
        return SampleConfig(
            temperature=self.temperature,
            top_k=self.top_k,
            mask_illegal=True,
            seed=self.seed,
        )


class SuiteResult(BaseModel):
    """Everything one evaluation produced; serialised verbatim as ``results.json``."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    suite: str
    checkpoint: str
    model_sha: str
    params: int
    date: str
    run_id: str | None = None
    legality: LegalityResult | None = None
    accuracy: AccuracyResult | None = None
    puzzles: PuzzleResult | None = None
    elo: EloResult | None = None
    delta_cp: float | None = None
    """Mean centipawn loss; measured by a later milestone, ``null`` until then."""
    diversity: float | None = None
    """Opening entropy over self-play games; measured by a later milestone."""
    notes: list[str] = []
    config: dict[str, Any] = {}


def config_path(suite: str) -> Path:
    """Where the YAML of a named suite lives inside the source tree."""
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; expected one of {', '.join(SUITES)}")
    return paths.package_root() / "configs" / "eval" / f"{suite}.yaml"


def load_suite(suite: str, config: Path | None = None) -> EvalConfig:
    """Load the suite config: an explicit path wins over the named one."""
    from rukh.config import load_yaml

    return load_yaml(config if config is not None else config_path(suite), EvalConfig)


def _positions(cfg: EvalConfig, tok: UciTokenizer, notes: list[str]) -> list[Any]:
    games = paths.resolve(cfg.games)
    if not games.is_file():
        notes.append(f"validation games not found at {games}: legality and accuracy skipped")
        return []
    return sample_positions(
        games,
        max(cfg.legality_positions, cfg.accuracy_positions),
        tok,
        seed=cfg.seed,
        pool=cfg.position_pool,
        block=cfg.block,
    )


def _elo(
    model: Any, tok: UciTokenizer, cfg: EvalConfig, cache: EvalCache | None, notes: list[str]
) -> EloResult | None:
    from rukh.engine import EngineNotFound

    try:
        records = play_rungs(
            model,
            tok,
            cfg.elo_rungs,
            cfg.elo_games,
            cfg.sampling(),
            move_time=cfg.elo_move_time,
            max_plies=cfg.elo_max_plies,
            cache=cache,
        )
    except EngineNotFound as exc:
        notes.append(f"Elo skipped: {exc}")
        return None
    if not records:
        notes.append("Elo skipped: no games were played")
        return None
    return estimate(records, samples=cfg.bootstrap, seed=cfg.seed)


def _metrics(result: SuiteResult) -> dict[str, float]:
    """Flat metrics for MLflow (only what was actually measured)."""
    metrics: dict[str, float] = {}
    if result.legality is not None:
        metrics["legality"] = result.legality.rate
    if result.accuracy is not None:
        metrics["top1"] = result.accuracy.top1
        metrics["top3"] = result.accuracy.top3
        for band in result.accuracy.bands:
            metrics[f"top1/{band.band}"] = band.top1
            metrics[f"top3/{band.band}"] = band.top3
    if result.puzzles is not None:
        metrics["puzzles"] = result.puzzles.rate
        for band in result.puzzles.bands:
            metrics[f"puzzles/{band.band}"] = band.rate
    if result.elo is not None:
        metrics["elo"] = result.elo.elo
        metrics["elo_ci_low"] = result.elo.ci_low
        metrics["elo_ci_high"] = result.elo.ci_high
    return metrics


def evaluate(
    ckpt: Path, cfg: EvalConfig, suite: str = "full", use_cache: bool = True
) -> SuiteResult:
    """Measure one checkpoint; missing inputs become notes, never exceptions."""
    from rukh.train import load_model

    ckpt = Path(ckpt)
    model, _payload = load_model(ckpt)
    tok = UciTokenizer()
    notes: list[str] = []
    cache = EvalCache(
        paths.resolve(cfg.cache_db) if use_cache else None, file_sha(ckpt), enabled=use_cache
    )
    try:
        positions = _positions(cfg, tok, notes)
        result = SuiteResult(
            stage=cfg.stage or ckpt.parent.name,
            suite=suite,
            checkpoint=ckpt.as_posix(),
            model_sha=cache.model_sha,
            params=model.num_params(non_embedding=False),
            date=datetime.now(UTC).date().isoformat(),
            legality=(
                legality(model, tok, positions[: cfg.legality_positions], cfg.sampling())
                if positions
                else None
            ),
            accuracy=(
                accuracy(model, tok, positions[: cfg.accuracy_positions]) if positions else None
            ),
            notes=notes,
            config=cfg.model_dump(mode="json"),
        )
        puzzle_path = paths.resolve(cfg.puzzles)
        if puzzle_path.is_file():
            items = load_puzzles(
                puzzle_path, cfg.puzzles_per_band, seed=cfg.seed, split=cfg.puzzle_split
            )
            result.puzzles = run_puzzles(model_source(model, tok), tok, items, cache=cache)
        else:
            notes.append(f"puzzles not found at {puzzle_path}: puzzle suite skipped")
        result.elo = _elo(model, tok, cfg, cache, notes)
        result.notes = notes  # pydantic copied the list at construction time
    finally:
        cache.close()
    return result


def track_result(result: SuiteResult, cfg: EvalConfig) -> str | None:
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


def run_suite(
    ckpt: Path, cfg: EvalConfig, suite: str = "full", use_cache: bool = True
) -> tuple[SuiteResult, ReportPaths]:
    """Evaluate, track and report: the whole of ``rukh eval`` in one call."""
    result = evaluate(ckpt, cfg, suite=suite, use_cache=use_cache)
    if cfg.track:
        result.run_id = track_result(result, cfg)
    report = write_report(
        result,
        paths.resolve(cfg.out_dir),
        paths.resolve(cfg.web_results) if cfg.web_results else None,
    )
    return result, report
