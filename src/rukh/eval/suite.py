"""The evaluation suite: one command that produces one row of the results table.

``run_suite`` loads a checkpoint, measures legality, next-move accuracy, puzzles and Elo, and
hands the result to ``report``. Every part is optional at run time: a missing validation
parquet, a missing puzzle parquet or a missing Stockfish binary becomes a note in the report
instead of a crash, because a partial evaluation that says what it could not measure is more
useful than no evaluation at all.

The suite runs on the same device the model was trained on (``pick_device`` by default, so CUDA
when it is there): a harness pinned to the CPU turns ten thousand forward passes and a few
hundred games into hours. The device that was used is recorded in the report.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rukh import paths
from rukh.config import BaseConfig
from rukh.eval.accuracy import AccuracyResult, accuracy
from rukh.eval.cache import EvalCache, config_sha, file_sha
from rukh.eval.diversity import DiversityResult, opening_diversity
from rukh.eval.elo import DEFAULT_RUNGS, EloResult, EloRung, estimate, play_rungs
from rukh.eval.legality import LegalityResult, legality, sample_positions
from rukh.eval.puzzles import (
    GAME_PREFIX,
    PuzzleResult,
    load_puzzles,
    model_source,
    run_puzzles,
)
from rukh.eval.report import ReportPaths, write_report
from rukh.infer import SampleConfig
from rukh.models import DecoderConfig
from rukh.tokenize.uci_vocab import UciTokenizer

log = logging.getLogger(__name__)

SUITES = ("full", "quick")
GOAL_LEGALITY = 0.99
"""The first acceptance criterion of ``GOAL.md`` for the decoder: legal moves without the mask.

Read on ``legality_argmax``, never on ``legality_sampled``: the bar is a property of the weights,
and the sampled rate is a property of the weights *and* of the temperature they were drawn at.
"""
GOAL_ELO = 1200.0
"""The second: at least 1200 estimated Elo, with its interval, against the Stockfish ladder."""
HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")
HUB_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
ELO_CAVEAT = (
    "the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal "
    "``Skill Level`` anchors rather than measured ratings, and Stockfish plays at {move_time:g} s "
    "per move, far below any setting ``UCI_Elo`` is calibrated for"
)
PUZZLE_PROMPT_NOTE = (
    "the puzzle parquet has no ``prefix_uci``, so every puzzle was prompted with its own "
    "solution line after ``<bos>``: a token sequence that is no game and does not start from "
    "the initial position. The rate is a floor, not a measurement, and is not comparable with a "
    "run scored from the real game prefix (rebuild the parquet with ``rukh data puzzles``)"
)
LEGALITY_DEFINITIONS = (
    "legality_argmax is the share of validation positions where the single most likely token is "
    "a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. "
    "legality_sampled draws the token the way the demo does (temperature {temperature:g}"
    "{top_k}) and is always the lower of the two."
)


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
    puzzles_use_header: bool = False
    """Prompt every puzzle with ``header_elo`` instead of the ratings its own players had.

    Off everywhere except the Elo sweep. Almost every puzzle carries real ratings, so with it off
    the puzzle rate is the same in every row of a sweep -- correct, and indistinguishable from
    evidence that the condition does nothing."""
    header_elo: int = 1_800
    """The Elo the model is asked to play at (``<wXXXX> <bXXXX>``).

    It was hardcoded to 1800 while the corpus was 1800+ Lichess and the choice cost nothing. With
    over half the corpus at 2400+ (D-064) it is a real knob: asking a model trained on masters to
    imitate an 1800 is asking it for its weaker mode, so the header belongs in the suite config
    and in the cache key rather than in a default argument.
    """
    elo_games: int = 100
    elo_rungs: list[EloRung] = Field(default_factory=lambda: list(DEFAULT_RUNGS))
    elo_move_time: float = 0.1
    elo_max_plies: int | None = None
    bootstrap: int = 1_000
    diversity_games: int = 200
    """Self-play openings for the diversity metric; 0 switches it off."""
    diversity_plies: int = 12
    diversity_temperature: float = 1.0
    """Read at its own temperature, not the suite's. The suite plays near-deterministically so
    that stages are comparable (D-047), and at that setting a decoder plays one single opening
    and the entropy is 0 for every model alike. Diversity only says something where there is a
    choice to make, so it gets the temperature that leaves one."""
    diversity_top_k: int | None = 20
    """And its own top-k, for the same reason and a sharper one: the suite reads at ``top_k: 1``,
    which is argmax whatever the temperature says. Overriding the temperature alone would look
    like a diversity measurement and be a second copy of the deterministic one."""
    temperature: float = 0.6
    top_k: int | None = 20
    block: int = 200
    seed: int = 42
    out_dir: str = "artifacts/eval"
    web_results: str | None = "artifacts/web/results.json"
    cache_db: str = "artifacts/eval/cache.sqlite"
    device: str | None = None
    """Where the model runs; ``None`` means ``rukh.train.pick_device()`` (CUDA when present)."""
    track: bool = True
    """Log the suite to the local MLflow store (the model card reads those metrics back)."""

    def cache_fields(self) -> dict[str, Any]:
        """The settings that change what a cached game or puzzle means."""
        fields: dict[str, Any] = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "elo_move_time": self.elo_move_time,
            "elo_max_plies": self.elo_max_plies,
            "elo_games": self.elo_games,
            "rungs": [rung.model_dump(mode="json") for rung in self.elo_rungs],
            "seed": self.seed,
            "block": self.block,
        }
        # Only when it is on. A cache key is a promise that two runs with the same key measured
        # the same thing, and a flag that is off *is* the behaviour every cached game was played
        # under; adding it unconditionally would throw away the 17 MB of games P2 and P3 paid for.
        if self.puzzles_use_header:
            fields["puzzles_use_header"] = True
        return fields

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
    device: str = "cpu"
    run_id: str | None = None
    legality_argmax: LegalityResult | None = None
    """The headline legality rate: the most likely token, unmasked."""
    legality_sampled: LegalityResult | None = None
    """Legality of a token drawn the way the demo draws it (temperature and top-k)."""
    accuracy: AccuracyResult | None = None
    puzzles: PuzzleResult | None = None
    elo: EloResult | None = None
    delta_cp: float | None = None
    """Mean centipawn loss; measured by a later milestone, ``null`` until then."""
    diversity: float | None = None
    """Normalised opening entropy over self-play games, 0 (always the same) to 1 (never twice)."""
    diversity_detail: DiversityResult | None = None
    """The whole measurement: distinct lines, the analytic first-move entropy and the
    temperature it was read at, which the normalised number alone would not carry."""
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
            header_elo=cfg.header_elo,
        )
    except EngineNotFound as exc:
        notes.append(f"Elo skipped: {exc}")
        return None
    if not records:
        notes.append("Elo skipped: no games were played")
        return None
    return estimate(records, samples=cfg.bootstrap, seed=cfg.seed)


def _metric_name(*parts: str) -> str:
    """A metric name MLflow will accept.

    Band names carry characters MLflow rejects outright (`<1800`, `2600+`), and it raises at the
    very end of a suite that has already played every game, so the whole run is lost to a label.
    Only alphanumerics, `_`, `-`, `.`, ` ` and `/` survive; everything else becomes `_`.
    """
    return "/".join(re.sub(r"[^A-Za-z0-9_.\- ]", "_", part) for part in parts)


def _metrics(result: SuiteResult) -> dict[str, float]:
    """Flat metrics for MLflow (only what was actually measured)."""
    metrics: dict[str, float] = {}
    if result.legality_argmax is not None:
        metrics["legality_argmax"] = result.legality_argmax.rate
    if result.legality_sampled is not None:
        metrics["legality_sampled"] = result.legality_sampled.rate
    if result.accuracy is not None:
        metrics["top1"] = result.accuracy.top1
        metrics["top3"] = result.accuracy.top3
        for band in result.accuracy.bands:
            metrics[_metric_name("top1", band.band)] = band.top1
            metrics[_metric_name("top3", band.band)] = band.top3
    if result.puzzles is not None:
        metrics["puzzles"] = result.puzzles.rate
        for band in result.puzzles.bands:
            metrics[_metric_name("puzzles", band.band)] = band.rate
    if result.elo is not None:
        metrics["elo"] = result.elo.elo
        if result.elo.ci_low is not None and result.elo.ci_high is not None:
            metrics["elo_ci_low"] = result.elo.ci_low
            metrics["elo_ci_high"] = result.elo.ci_high
        if result.elo.elo_lower is not None:
            metrics["elo_lower_bound"] = result.elo.elo_lower
        if result.elo.elo_upper is not None:
            metrics["elo_upper_bound"] = result.elo.elo_upper
    return metrics


def is_hub_id(spec: str) -> bool:
    """Whether ``spec`` looks like a Hub repository id (``owner/name``) rather than a path."""
    text = str(spec)
    if Path(text).expanduser().is_file() or text.endswith(".pt"):
        return False
    return HUB_ID.fullmatch(text) is not None


def hub_checkpoint(repo_id: str, out_dir: Path | None = None) -> Path:
    """Download a published model and write it back as a checkpoint the harness can read.

    A Hub repository holds ``config.json`` and ``model.safetensors`` (or the torch pickle when
    safetensors was not installed at publication time), not a training checkpoint, so the two are
    folded into the ``{model_state, model_cfg, step}`` payload every other entry point expects.
    Nothing here runs while the command line is being parsed: a Hub id is validated by its shape.
    """
    import torch
    from huggingface_hub import hf_hub_download

    directory = out_dir if out_dir is not None else paths.resolve("artifacts/hub")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (repo_id.replace("/", "__") + ".pt")
    config = json.loads(Path(hf_hub_download(repo_id, "config.json")).read_text(encoding="utf-8"))
    state: dict[str, Any] = {}
    for name in HUB_WEIGHTS:
        try:
            weights = Path(hf_hub_download(repo_id, name))
        except Exception as exc:  # noqa: BLE001 - a missing file only means "try the next one"
            log.debug("%s has no %s (%s)", repo_id, name, exc)
            continue
        if name.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = dict(load_file(str(weights)))
        else:
            state = dict(torch.load(weights, map_location="cpu", weights_only=True))
        break
    if not state:
        raise FileNotFoundError(f"{repo_id} holds none of {', '.join(HUB_WEIGHTS)}")
    model_cfg = {key: value for key, value in config.items() if key in DecoderConfig.model_fields}
    torch.save(
        {
            "step": int(config.get("step") or 0),
            "model_state": state,
            "opt_state": None,
            "cfg": {},
            "model_cfg": model_cfg,
            "vocab_hash": config.get("vocab_hash"),
            "data_manifest_sha": config.get("data_manifest_sha"),
            "git_sha": config.get("git_sha"),
            "best_val": None,
        },
        target,
    )
    return target


def resolve_model(spec: str | Path) -> Path:
    """A checkpoint path from either a local file or a Hub id (downloaded on demand)."""
    text = str(spec)
    if is_hub_id(text):
        return hub_checkpoint(text)
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is neither a checkpoint nor a Hub id (owner/name)")
    return path


def _legality_note(cfg: EvalConfig) -> str:
    top_k = "" if cfg.top_k is None else f", top-k {cfg.top_k}"
    return LEGALITY_DEFINITIONS.format(temperature=cfg.temperature, top_k=top_k)


def evaluate(
    ckpt: Path | str,
    cfg: EvalConfig,
    suite: str = "full",
    use_cache: bool = True,
    device: str | None = None,
) -> SuiteResult:
    """Measure one checkpoint; missing inputs become notes, never exceptions."""
    from rukh.train import load_model, pick_device

    ckpt = resolve_model(ckpt)
    where = device or cfg.device or pick_device()
    model, _payload = load_model(ckpt, map_location=where)
    model = model.to(where).eval()
    log.info("evaluating %s on %s", ckpt, where)
    tok = UciTokenizer()
    notes: list[str] = [_legality_note(cfg)]
    cache = EvalCache(
        paths.resolve(cfg.cache_db) if use_cache else None,
        file_sha(ckpt),
        enabled=use_cache,
        config_sha=config_sha(cfg.cache_fields()),
    )
    try:
        positions = _positions(cfg, tok, notes)
        for_legality = positions[: cfg.legality_positions]
        result = SuiteResult(
            stage=cfg.stage or ckpt.parent.name,
            suite=suite,
            checkpoint=ckpt.as_posix(),
            model_sha=cache.model_sha,
            params=model.num_params(non_embedding=False),
            date=datetime.now(UTC).date().isoformat(),
            device=str(where),
            legality_argmax=(
                legality(model, tok, for_legality, cfg.sampling(), mode="argmax")
                if positions
                else None
            ),
            legality_sampled=(
                legality(model, tok, for_legality, cfg.sampling(), mode="sampled")
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
            result.puzzles = run_puzzles(
                model_source(model, tok),
                tok,
                items,
                cache=cache,
                header_elo=cfg.header_elo,
                force_header=cfg.puzzles_use_header,
            )
            if result.puzzles.prompt_style != GAME_PREFIX:
                notes.append(PUZZLE_PROMPT_NOTE)
        else:
            notes.append(f"puzzles not found at {puzzle_path}: puzzle suite skipped")
        if cfg.diversity_games:
            detail = opening_diversity(
                model,
                tok,
                games=cfg.diversity_games,
                plies=cfg.diversity_plies,
                sampling=cfg.sampling().model_copy(
                    update={
                        "temperature": cfg.diversity_temperature,
                        "top_k": cfg.diversity_top_k,
                    }
                ),
                header_elo=cfg.header_elo,
                seed=cfg.seed,
            )
            result.diversity_detail = detail
            result.diversity = detail.normalised
        result.elo = _elo(model, tok, cfg, cache, notes)
        _elo_notes(result.elo, cfg, notes)
        result.notes = notes  # pydantic copied the list at construction time
    finally:
        cache.close()
    return result


def _elo_notes(elo: EloResult | None, cfg: EvalConfig, notes: list[str]) -> None:
    """Everything the Elo number has to be read with, in words."""
    if elo is None:
        return
    notes.append(ELO_CAVEAT.format(move_time=cfg.elo_move_time))
    if elo.cut:
        notes.append(
            f"{elo.cut} of {elo.games} games hit the context limit and were adjudicated "
            f"({elo.adjudicated} of them) instead of being scored as draws"
        )
    if elo.separated:
        notes.append(
            "every game went the same way, so the rating is not identified: the report gives a "
            "one-sided bound instead of a 95 % interval"
        )


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
    ckpt: Path | str,
    cfg: EvalConfig,
    suite: str = "full",
    use_cache: bool = True,
    device: str | None = None,
) -> tuple[SuiteResult, ReportPaths]:
    """Evaluate, track and report: the whole of ``rukh eval`` in one call."""
    result = evaluate(ckpt, cfg, suite=suite, use_cache=use_cache, device=device)
    if cfg.track:
        result.run_id = track_result(result, cfg)
    report = write_report(
        result,
        paths.resolve(cfg.out_dir),
        paths.resolve(cfg.web_results) if cfg.web_results else None,
    )
    return result, report
