"""``rukh eval nightly``: the whole results table from the published models, with one command.

Twelve decoder rows and four encoder rows were written one ``rukh eval`` at a time over five
milestones, each with the config of its day. This walks the catalogue of ``rukh.hub`` -- every
model the course published, with the stage name its row carries and the suite that measures it
-- and evaluates each one under the **same** config, pulling from the Hub whatever is not on
disk. What the caches already hold is not replayed, so a nightly after a nightly is minutes;
the first one is hours of engine.

It writes three things: the rows of ``artifacts/web/results.json`` it measured (upserted by
stage, the rows it did not measure are left alone), ``docs/benchmarks.md`` rendered from the
whole table, and ``artifacts/eval/nightly.json``: which stage came from which checkpoint with
which sha, how long each took and what went wrong. A table is reproducible when the last file
says how it was made.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rukh import paths
from rukh.hub import Artefact, catalogue, pull
from rukh.train.checkpoint import resolve_run

log = logging.getLogger(__name__)

NIGHTLY_FILE = "artifacts/eval/nightly.json"
BENCHMARKS_FILE = "docs/benchmarks.md"
MERGED_DIR = "artifacts/eval/merged"


class NightlyRecord(BaseModel):
    """One stage of the run: what was measured, from what, and how it went."""

    model_config = ConfigDict(extra="forbid")

    name: str
    stage: str
    measure: str
    checkpoint: str | None = None
    model_sha: str | None = None
    seconds: float = 0.0
    status: str = "planned"
    """``planned`` (dry run), ``measured``, ``skipped`` or ``failed``."""
    detail: str | None = None


class NightlyReport(BaseModel):
    """What one nightly did; serialised as ``artifacts/eval/nightly.json``."""

    model_config = ConfigDict(extra="forbid")

    date: str
    config: str
    encoder_config: str | None
    records: list[NightlyRecord]
    results: str
    benchmarks: str | None
    seconds: float


def plan(only: set[str] | None = None) -> list[Artefact]:
    """The catalogue entries a nightly measures, in course order."""
    wanted = [artefact for artefact in catalogue() if artefact.measure != "none"]
    if only:
        wanted = [a for a in wanted if a.name in only or (a.stage or "") in only]
    return wanted


def _run_decoder(checkpoint: Path, config: Path | None, stage: str, use_cache: bool, device):
    from rukh.eval.suite import load_suite, run_suite

    cfg = load_suite("full", config).model_copy(update={"stage": stage})
    return run_suite(checkpoint, cfg, suite="full", use_cache=use_cache, device=device)


def _run_encoder(checkpoint: Path, config: Path | None, stage: str, use_cache: bool, device):
    from rukh.eval.encoder import load_encoder_suite, run_encoder_suite

    cfg = load_encoder_suite(config).model_copy(update={"stage": stage})
    return run_encoder_suite(checkpoint, cfg, use_cache=use_cache, device=device)


def _run_qwen(adapter_dir: Path, config: Path | None, stage: str, use_cache: bool, device):
    from rukh.eval.qwen_suite import run_qwen_suite
    from rukh.eval.suite import load_suite

    cfg = load_suite("full", config)
    return run_qwen_suite(adapter_dir, cfg, stage=stage, use_cache=use_cache, device=device)


def merged_checkpoint(base: Path, adapter_dir: Path, out: Path) -> Path:
    """The base with the adapter folded in, as an ordinary checkpoint the suite can open.

    A style adapter is 1.6 MB of factors and measures nothing on its own; the row of the table
    is the base *with* it. ``load_adapter`` wraps the adapted matrices and reads the factors,
    ``merge_lora`` folds them back into plain ``nn.Linear`` weights, and what is saved has the
    state dict of a normal decoder -- the same thing the LoRA training run wrote as ``best.pt``.
    """
    from rukh.models.lora import ADAPTER_FILE, load_adapter, merge_lora
    from rukh.train.checkpoint import load_model, save_checkpoint

    model, payload = load_model(base)
    load_adapter(model, adapter_dir / ADAPTER_FILE)
    merge_lora(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    return save_checkpoint(
        out,
        step=int(payload.get("step") or 0),
        model=model,
        optimizer=None,
        cfg={"merged_from": base.as_posix(), "adapter": adapter_dir.as_posix()},
        model_cfg=payload["model_cfg"],
        vocab_hash=payload.get("vocab_hash"),
        data_manifest_sha=payload.get("data_manifest_sha"),
        git_sha=payload.get("git_sha"),
    )


def _checkpoint_of(artefact: Artefact, pull_missing: bool) -> Path:
    """Where the artefact is on disk, pulling it from the Hub when it is not and allowed."""
    target = resolve_run(paths.resolve(artefact.target))
    if not target.exists() and pull_missing:
        target = pull(artefact)
    if not target.exists():
        raise FileNotFoundError(
            f"{artefact.name}: {target} is not on disk (rukh pull {artefact.name})"
        )
    if artefact.base is not None:
        base = resolve_run(paths.resolve(artefact.base))
        if not base.exists() and pull_missing:
            from rukh.hub import lookup

            base = pull(lookup("medium-v4"))
        if not base.exists():
            raise FileNotFoundError(f"{artefact.name}: base {base} is not on disk")
        return merged_checkpoint(base, target, paths.resolve(MERGED_DIR) / f"{artefact.name}.pt")
    return target


def run_nightly(
    config: Path | None = None,
    encoder_config: Path | None = None,
    only: set[str] | None = None,
    dry_run: bool = False,
    pull_missing: bool = True,
    use_cache: bool = True,
    device: str | None = None,
    results: Path | None = None,
    benchmarks: Path | None = None,
    out: Path | None = None,
) -> NightlyReport:
    """Measure every catalogued stage under one config and rewrite the table around them."""
    from rukh.eval.cache import file_sha
    from rukh.eval.report import write_benchmarks

    started = time.perf_counter()
    records: list[NightlyRecord] = []
    for artefact in plan(only):
        record = NightlyRecord(
            name=artefact.name, stage=artefact.stage or artefact.name, measure=artefact.measure
        )
        records.append(record)
        if dry_run:
            record.checkpoint = paths.resolve(artefact.target).as_posix()
            continue
        tick = time.perf_counter()
        try:
            checkpoint = _checkpoint_of(artefact, pull_missing)
            record.checkpoint = checkpoint.as_posix()
            if checkpoint.is_file():
                record.model_sha = file_sha(checkpoint)
            log.info("nightly: %s as %s from %s", artefact.name, record.stage, checkpoint)
            if artefact.measure == "decoder":
                _run_decoder(checkpoint, config, record.stage, use_cache, device)
            elif artefact.measure == "encoder":
                _run_encoder(checkpoint, encoder_config, record.stage, use_cache, device)
            elif artefact.measure == "qwen":
                _run_qwen(checkpoint, config, record.stage, use_cache, device)
            record.status = "measured"
        except Exception as exc:  # noqa: BLE001 - one stage failing must not lose the others
            log.exception("nightly: %s failed", artefact.name)
            record.status = "failed"
            record.detail = f"{type(exc).__name__}: {exc}"
        record.seconds = round(time.perf_counter() - tick, 1)

    results_path = results or paths.resolve("artifacts/web/results.json")
    benchmarks_path: Path | None = None
    if not dry_run and results_path.is_file():
        benchmarks_path = benchmarks or paths.resolve(BENCHMARKS_FILE)
        if benchmarks_path.is_file():
            write_benchmarks(results_path, benchmarks_path)
        else:
            benchmarks_path = None
    report = NightlyReport(
        date=datetime.now(UTC).date().isoformat(),
        config=(config or Path("configs/eval/full.yaml")).as_posix(),
        encoder_config=encoder_config.as_posix() if encoder_config else None,
        records=records,
        results=results_path.as_posix(),
        benchmarks=benchmarks_path.as_posix() if benchmarks_path else None,
        seconds=round(time.perf_counter() - started, 1),
    )
    if not dry_run:
        target = out or paths.resolve(NIGHTLY_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def render_plan(report: NightlyReport) -> str:
    """One line per stage, for the terminal."""
    lines = [f"{'stage':<32}{'measure':<9}{'status':<10}{'seconds':>8}  checkpoint"]
    for record in report.records:
        lines.append(
            f"{record.stage:<32}{record.measure:<9}{record.status:<10}{record.seconds:>8.1f}  "
            f"{record.checkpoint or '-'}" + (f"  ({record.detail})" if record.detail else "")
        )
    return "\n".join(lines)


__all__: list[str] = [
    "BENCHMARKS_FILE",
    "MERGED_DIR",
    "NIGHTLY_FILE",
    "NightlyRecord",
    "NightlyReport",
    "merged_checkpoint",
    "plan",
    "render_plan",
    "run_nightly",
]
