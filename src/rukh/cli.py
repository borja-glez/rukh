"""Command-line interface: info, data, train, play, eval, export, publish, engine, mlflow.

Every command is a thin shell over the library: parse options, call one function, print.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Annotated

import typer

from rukh import __version__

app = typer.Typer(
    name="rukh",
    help="Rukh: chess language model toolkit.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Datasets: fetch, convert, tokenize and publish.", no_args_is_help=True)
app.add_typer(data_app, name="data")
mlflow_app = typer.Typer(help="Local MLflow tracking.", no_args_is_help=True)
app.add_typer(mlflow_app, name="mlflow")
engine_app = typer.Typer(help="Stockfish engine utilities.", no_args_is_help=True)
app.add_typer(engine_app, name="engine")
publish_app = typer.Typer(help="Publish trained models to the Hub.", no_args_is_help=True)
app.add_typer(publish_app, name="publish")
train_app = typer.Typer(
    help="Train a model: the decoder by default, a subcommand for the encoder.",
    invoke_without_command=True,
)
app.add_typer(train_app, name="train")
eval_app = typer.Typer(
    help="Evaluate a model: the decoder by default, a subcommand for the encoder.",
    invoke_without_command=True,
)
app.add_typer(eval_app, name="eval")
encoder_app = typer.Typer(help="Encoder utilities: position embeddings.", no_args_is_help=True)
app.add_typer(encoder_app, name="encoder")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"rukh {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the package version and exit.",
        ),
    ] = False,
) -> None:
    """Rukh: chess language model toolkit."""


@app.command()
def info(
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON only.")] = False,
) -> None:
    """Report Python, torch, CUDA, GPU, Stockfish and project directories."""
    from rukh.env import EnvReport, collect

    report = collect()
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    width = max(len(key) for key in EnvReport.model_fields)
    for key, value in report.model_dump().items():
        typer.echo(f"{key:<{width}}  {value if value is not None else 'null'}")


@data_app.command("fetch")
def data_fetch(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True, help="YAML config."),
    ],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan; touch neither network nor disk.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Refetch months whose parquet is already on disk.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the plan as JSON only.")] = False,
) -> None:
    """Fetch filtered Lichess games as parquet files plus a manifest."""
    from rukh.config import load_yaml
    from rukh.data.fetch import FetchConfig, run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = load_yaml(config, FetchConfig)
    fetch_plan = run(cfg, dry_run=dry_run, overwrite=overwrite)
    if as_json:
        typer.echo(fetch_plan.model_dump_json(indent=2))
        return
    typer.echo(f"dataset:  {cfg.dataset}")
    typer.echo(f"months:   {', '.join(fetch_plan.months)}")
    typer.echo(f"mode:     {'dry-run (nothing written)' if dry_run else 'fetched'}")
    typer.echo("query:")
    for line in fetch_plan.query.splitlines():
        typer.echo(f"  {line}")
    typer.echo(f"out_dir:  {fetch_plan.out_dir}")
    typer.echo("outputs:")
    for path in fetch_plan.out_paths:
        typer.echo(f"  {path}")
    typer.echo(f"manifest: {fetch_plan.manifest_path}")


PipelineOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Pipeline YAML (default: configs/data/pipeline.yaml).",
    ),
]


def _echo_written(written: list[str]) -> None:
    typer.echo("written:")
    for path in written:
        typer.echo(f"  {path}")


@data_app.command("uci")
def data_uci(
    config: PipelineOption = None,
    workers: Annotated[
        int | None, typer.Option("--workers", help="Override the pool size from the config.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the manifest as JSON only.")
    ] = False,
) -> None:
    """Convert fetched SAN movetext to legal UCI games, one parquet per month."""
    from rukh.data.pipeline import load_pipeline
    from rukh.data.uci import run

    cfg = load_pipeline(config).uci
    if workers is not None:
        cfg = cfg.model_copy(update={"workers": workers})
    manifest = run(cfg)
    if as_json:
        typer.echo(manifest.model_dump_json(indent=2))
        return
    typer.echo(f"months:   {', '.join(manifest.months)}")
    for month, count in manifest.counts.items():
        typer.echo(f"  {month}: {count} games kept")
    typer.echo(f"manifest: {cfg.out_dir}/manifest.json")


@data_app.command("tokenize")
def data_tokenize(
    config: PipelineOption = None,
    scheme: Annotated[
        str, typer.Option("--scheme", help="Tokenization scheme: uci, san or bpe.")
    ] = "uci",
    export_fixture: Annotated[
        bool,
        typer.Option(
            "--export-fixture",
            help="Write artifacts/tokenizer/vocab.json and fixtures/games.json.",
        ),
    ] = False,
    stats: Annotated[
        bool,
        typer.Option("--stats", help="Write artifacts/web/tokenizer-stats.json for all schemes."),
    ] = False,
    games: Annotated[
        Path | None,
        typer.Option(
            "--games",
            exists=True,
            dir_okay=False,
            help="UCI games parquet for BPE training and statistics (default from config).",
        ),
    ] = None,
    pack: Annotated[
        bool,
        typer.Option("--pack", help="Pack train/val months into data/tokens/<scheme>/."),
    ] = False,
) -> None:
    """Build tokenizer artifacts, the parity fixture, statistics and packed token streams."""
    from rukh.data.pipeline import load_pipeline
    from rukh.tokenize.run import SCHEMES, run

    if scheme not in SCHEMES:
        typer.echo(f"error: --scheme must be one of {', '.join(SCHEMES)}", err=True)
        raise typer.Exit(code=2)
    cfg = load_pipeline(config).tokenize
    report = run(
        cfg, scheme=scheme, export_fixture=export_fixture, stats=stats, games=games, pack=pack
    )
    typer.echo(f"scheme:     {report.scheme}")
    typer.echo(f"vocab_size: {report.vocab_size}")
    _echo_written(report.written)


def _echo_manifest(manifest: object, as_json: bool, out_dir: str) -> None:
    from rukh.data.manifest import Manifest

    assert isinstance(manifest, Manifest)
    if as_json:
        typer.echo(manifest.model_dump_json(indent=2))
        return
    typer.echo(f"dataset:  {manifest.dataset}")
    if manifest.months:
        typer.echo(f"months:   {', '.join(manifest.months)}")
    typer.echo("counts:")
    for key, value in manifest.counts.items():
        typer.echo(f"  {key}: {value}")
    typer.echo("files:")
    for file in manifest.files:
        typer.echo(f"  {file.path} ({file.bytes} bytes)")
    typer.echo(f"manifest: {out_dir}/manifest.json")


JsonOption = Annotated[bool, typer.Option("--json", help="Print the manifest as JSON.")]


@data_app.command("positions")
def data_positions(config: PipelineOption = None, as_json: JsonOption = False) -> None:
    """Walk sampled games into deduplicated 4-field FENs with phase labels."""
    from rukh.data.pipeline import load_pipeline
    from rukh.data.positions import run

    cfg = load_pipeline(config).positions
    _echo_manifest(run(cfg), as_json, cfg.out_dir)


@data_app.command("evals")
def data_evals(
    config: PipelineOption = None,
    files: Annotated[
        str | None,
        typer.Option("--files", help="Remote file indices, e.g. 0-19, 3 or 0,2,5-7 (default all)."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Semi-join the positions with the remote Stockfish evaluations (resumable per file)."""
    from rukh.data.evals import run
    from rukh.data.pipeline import load_pipeline

    cfg = load_pipeline(config).evals
    _echo_manifest(run(cfg, files=files), as_json, cfg.out_dir)


@data_app.command("puzzles")
def data_puzzles(config: PipelineOption = None, as_json: JsonOption = False) -> None:
    """Filter Lichess puzzles and split them by difficulty band with a fixed seed."""
    from rukh.data.pipeline import load_pipeline
    from rukh.data.puzzles import run

    cfg = load_pipeline(config).puzzles
    _echo_manifest(run(cfg), as_json, cfg.out_dir)


@data_app.command("pairs")
def data_pairs(config: PipelineOption = None, as_json: JsonOption = False) -> None:
    """Build phase-balanced DPO pairs from the multi-PV evaluations."""
    from rukh.data.pairs import run
    from rukh.data.pipeline import load_pipeline

    cfg = load_pipeline(config).pairs
    _echo_manifest(run(cfg), as_json, cfg.out_dir)


@data_app.command("elite")
def data_elite(config: PipelineOption = None, as_json: JsonOption = False) -> None:
    """Download the Lichess Elite Database months and convert them to UCI."""
    from rukh.data.elite import run
    from rukh.data.pipeline import load_pipeline

    cfg = load_pipeline(config).elite
    _echo_manifest(run(cfg), as_json, cfg.out_dir)


@data_app.command("elo-bins")
def data_elo_bins(config: PipelineOption = None, as_json: JsonOption = False) -> None:
    """Sample up to n_per_bin games per 100-Elo bin of the average rating."""
    from rukh.data.elo_bins import run
    from rukh.data.pipeline import load_pipeline

    cfg = load_pipeline(config).elo_bins
    _echo_manifest(run(cfg), as_json, cfg.out_dir)


@data_app.command("pgn-text")
def data_pgn_text(config: PipelineOption = None, as_json: JsonOption = False) -> None:
    """Render the same games as PGN text, for the general-model comparison of M4."""
    from rukh.data.pgn_text import PgnTextConfig, build
    from rukh.data.pipeline import load_pipeline

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = load_pipeline(config).pgn_text if config is not None else PgnTextConfig()
    try:
        manifest = build(cfg)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _echo_manifest(manifest, as_json, cfg.out_dir)


@data_app.command("style")
def data_style(config: PipelineOption = None, as_json: JsonOption = False) -> None:
    """Cut one parquet per style: the games a LoRA adapter is trained to sound like."""
    from rukh.data.pipeline import load_pipeline
    from rukh.data.style import build_styles

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = load_pipeline(config).style
    try:
        manifests = build_styles(cfg)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo("[" + ", ".join(m.model_dump_json(indent=2) for m in manifests.values()) + "]")
        return
    typer.echo(f"out_dir:  {cfg.out_dir}")
    typer.echo("styles:")
    for name, manifest in manifests.items():
        typer.echo(f"  {name}: {manifest.counts[name]:,} games")


@data_app.command("publish")
def data_publish(
    name: Annotated[
        str | None, typer.Option("--name", help="Registry entry, e.g. rukh-games-1800.")
    ] = None,
    config: PipelineOption = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Render the card and list files; no network.")
    ] = False,
    list_names: Annotated[bool, typer.Option("--list", help="List the registry and exit.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Render the dataset card and upload one dataset (or the tokenizer) to the Hub."""
    from rukh.data.pipeline import load_pipeline
    from rukh.data.publish import DATASETS, publish

    if list_names:
        for spec in DATASETS.values():
            typer.echo(f"{spec.name:<22} {spec.repo_type:<8} {spec.local_dir}")
        return
    if name is None:
        typer.echo("error: --name is required (see --list)", err=True)
        raise typer.Exit(code=2)
    cfg = load_pipeline(config).publish
    try:
        result = publish(name, cfg, dry_run=dry_run)
    except (KeyError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"repo:     {result.repo_id} ({result.repo_type})")
    typer.echo(f"mode:     {'dry-run (card staged, nothing uploaded)' if dry_run else 'uploaded'}")
    typer.echo(f"rows:     {result.rows}")
    typer.echo(f"card:     {result.card_path}")
    typer.echo("files:")
    for path in result.files:
        typer.echo(f"  {path}")


@train_app.callback(invoke_without_command=True)
def train_cmd(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, readable=True, help="Training YAML config."
        ),
    ] = None,
    model_preset: Annotated[
        str | None, typer.Option("--preset", help="Override the preset: tiny, small or medium.")
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume", exists=True, dir_okay=False, readable=True, help="Checkpoint to continue."
        ),
    ] = None,
    max_steps: Annotated[
        int | None, typer.Option("--max-steps", help="Override max_steps from the config.")
    ] = None,
) -> None:
    """Train a MoveDecoder from a packed token stream, logging the run to MLflow.

    ``rukh train --config ...`` is the decoder, exactly as it always was; the subcommands train
    the other models (``rukh train encoder``).
    """
    from rukh.config import load_yaml
    from rukh.models import PRESETS
    from rukh.train import TrainConfig, train

    if ctx.invoked_subcommand is not None:
        return
    if config is None:
        typer.echo("error: --config is required (see rukh train --help)", err=True)
        raise typer.Exit(code=2)
    cfg = load_yaml(config, TrainConfig)
    if model_preset is not None:
        if model_preset not in PRESETS:
            typer.echo(f"error: --preset must be one of {', '.join(PRESETS)}", err=True)
            raise typer.Exit(code=2)
        cfg = cfg.model_copy(update={"preset": model_preset, "model": None})
    if max_steps is not None:
        cfg = cfg.model_copy(update={"max_steps": max_steps})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    try:
        checkpoint = train(cfg, resume=resume)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"preset:     {cfg.preset}")
    typer.echo(f"steps:      {cfg.max_steps}")
    typer.echo(f"checkpoint: {checkpoint}")


@train_app.command("encoder")
def train_encoder_cmd(
    config: Annotated[
        Path,
        typer.Option(
            "--config", exists=True, dir_okay=False, readable=True, help="Training YAML config."
        ),
    ],
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume", exists=True, dir_okay=False, readable=True, help="Checkpoint to continue."
        ),
    ] = None,
    max_steps: Annotated[
        int | None, typer.Option("--max-steps", help="Override max_steps from the config.")
    ] = None,
) -> None:
    """Pretrain the PositionEncoder with masked move modeling, logging the run to MLflow."""
    from rukh.config import load_yaml
    from rukh.train import MmmConfig, train_mmm

    cfg = load_yaml(config, MmmConfig)
    if max_steps is not None:
        cfg = cfg.model_copy(update={"max_steps": max_steps})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    try:
        checkpoint = train_mmm(cfg, resume=resume)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"input:      {cfg.input}")
    typer.echo(
        f"masking:    {cfg.masking.prob:.0%} at "
        f"{cfg.masking.mask_ratio:.0%}/{cfg.masking.random_ratio:.0%}/"
        f"{cfg.masking.keep_ratio:.0%}"
    )
    typer.echo(f"steps:      {cfg.max_steps}")
    typer.echo(f"checkpoint: {checkpoint}")


@train_app.command("qwen")
def train_qwen_cmd(
    config: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False, readable=True, help="YAML config."),
    ] = None,
    device: Annotated[str | None, typer.Option("--device", help="Where to run.")] = None,
) -> None:
    """QLoRA a general language model on PGN text, for the comparison table of M4.

    Four-bit is attempted and not required: at 0.6 B parameters on a 32 GB card it saves nothing
    that matters, so if `bitsandbytes` will not load the run falls back to bf16 and says so in
    its report instead of claiming a technique it did not use.
    """
    from rukh.config import load_yaml
    from rukh.train.qwen import QwenConfig, train_qwen

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = load_yaml(config, QwenConfig) if config is not None else QwenConfig()
    try:
        report = train_qwen(cfg, device=device)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"model:     {report.model}")
    typer.echo(f"precision: {'4-bit NF4' if report.four_bit else 'bf16'}")
    if report.fallback_reason:
        typer.echo(f"fallback:  {report.fallback_reason}")
    typer.echo(
        f"trainable: {report.trainable_params:,} of {report.total_params:,} "
        f"({100 * report.trainable_share:.3f} %)"
    )
    typer.echo(f"samples:   {report.train_samples:,} over {report.steps:,} steps")
    if report.final_loss is not None:
        typer.echo(f"loss:      {report.final_loss:.4f}")
    if report.weights_memory_mb is not None:
        typer.echo(f"weights:   {report.weights_memory_mb:,.0f} MB on the device")
    if report.peak_memory_mb is not None:
        typer.echo(f"peak mem:  {report.peak_memory_mb:,.0f} MB during training")
    typer.echo(f"adapter:   {report.adapter_dir}")


@train_app.command("heads")
def train_heads_cmd(
    config: Annotated[
        Path,
        typer.Option(
            "--config", exists=True, dir_okay=False, readable=True, help="Training YAML config."
        ),
    ],
    mode: Annotated[
        str | None, typer.Option("--mode", help="Override the mode: probe, last-n or full.")
    ] = None,
    fraction: Annotated[
        float | None, typer.Option("--fraction", help="Train on this fraction of the labels.")
    ] = None,
    curve: Annotated[
        bool, typer.Option("--curve", help="Run the whole label-count curve instead of one run.")
    ] = False,
    max_steps: Annotated[
        int | None, typer.Option("--max-steps", help="Override max_steps from the config.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the results as JSON only.")
    ] = False,
) -> None:
    """Fine-tune the value, blunder and result heads over the pretrained encoder."""
    import json

    from rukh.config import load_yaml
    from rukh.train import HeadsConfig, label_curve, train_heads

    cfg = load_yaml(config, HeadsConfig)
    if mode is not None:
        if mode not in ("probe", "last-n", "full"):
            typer.echo("error: --mode must be probe, last-n or full", err=True)
            raise typer.Exit(code=2)
        cfg = cfg.model_copy(update={"mode": mode})
    if fraction is not None:
        cfg = cfg.model_copy(update={"fraction": fraction})
    if max_steps is not None:
        cfg = cfg.model_copy(update={"max_steps": max_steps})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    try:
        results = label_curve(cfg) if curve else [train_heads(cfg)]
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(json.dumps([result.model_dump() for result in results], indent=2))
        return
    typer.echo(f"mode:       {cfg.mode}")
    for result in results:
        typer.echo(f"  {result.fraction:>5.0%} of the labels ({result.train_labels} rows)")
        for key, value in result.metrics.items():
            typer.echo(f"    {key:<20} {value:.4f}")
        typer.echo(f"    checkpoint           {result.checkpoint}")


@app.command("play")
def play_cmd(
    ckpt: Annotated[
        Path,
        typer.Option(
            "--ckpt", exists=True, dir_okay=False, readable=True, help="Checkpoint to play with."
        ),
    ],
    games: Annotated[int, typer.Option("--games", help="Number of games to play.")] = 1,
    opponent: Annotated[
        str, typer.Option("--opponent", help="Opponent: random or stockfish.")
    ] = "random",
    elo: Annotated[int, typer.Option("--elo", help="UCI_Elo of the Stockfish opponent.")] = 1400,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature (0 = argmax).")
    ] = 0.6,
    top_k: Annotated[int, typer.Option("--top-k", help="Top-k truncation; 0 disables it.")] = 20,
    no_mask: Annotated[
        bool, typer.Option("--no-mask", help="Sample without the legality mask (measures it).")
    ] = False,
    seed: Annotated[int, typer.Option("--seed", help="Seed for sampling and the opponent.")] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Print the games as JSON only.")] = False,
) -> None:
    """Play games between a trained decoder and an opponent, with or without the legality mask."""
    import json

    from rukh.infer import RandomOpponent, SampleConfig, StockfishOpponent, play_game
    from rukh.tokenize.uci_vocab import UciTokenizer
    from rukh.train import load_model

    if opponent not in ("random", "stockfish"):
        typer.echo("error: --opponent must be random or stockfish", err=True)
        raise typer.Exit(code=2)
    model, _ = load_model(ckpt)
    tok = UciTokenizer()
    cfg = SampleConfig(
        temperature=temperature,
        top_k=top_k or None,
        mask_illegal=not no_mask,
        seed=seed,
    )
    rival = RandomOpponent(seed) if opponent == "random" else StockfishOpponent(elo=elo)
    try:
        results = [
            play_game(model, tok, rival, cfg, model_color=index % 2 == 0) for index in range(games)
        ]
    finally:
        if isinstance(rival, StockfishOpponent):
            rival.close()
    if as_json:
        typer.echo(json.dumps([result.model_dump() for result in results], indent=2))
        return
    typer.echo(f"opponent: {opponent}")
    typer.echo(f"mask:     {'on' if cfg.mask_illegal else 'off'}")
    for index, result in enumerate(results):
        typer.echo(
            f"  game {index + 1}: {result.result:<7} {result.plies:>3} plies  "
            f"{result.illegal_proposals} illegal  ({result.termination})"
        )
    typer.echo(f"illegal:  {sum(r.illegal_proposals for r in results)} proposals")


@eval_app.callback(invoke_without_command=True)
def eval_cmd(
    ctx: typer.Context,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Checkpoint path or Hub id (owner/name) to evaluate."),
    ] = None,
    suite: Annotated[str, typer.Option("--suite", help="Suite name: full or quick.")] = "full",
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, readable=True, help="Suite YAML override."
        ),
    ] = None,
    stage: Annotated[
        str | None, typer.Option("--stage", help="Row name in the results table.")
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Recompute every game and puzzle.")
    ] = False,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Where to run: cuda, cpu... (default: the training device)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON only.")] = False,
) -> None:
    """Measure legality, next-move accuracy, puzzles and Elo, and write the report.

    ``rukh eval --model ...`` is the decoder, exactly as it always was; the subcommands
    evaluate the other models (``rukh eval encoder``).
    """
    from rukh.eval import load_suite, run_suite
    from rukh.eval.report import elo_line
    from rukh.eval.suite import SUITES, is_hub_id

    if ctx.invoked_subcommand is not None:
        return
    if model is None:
        typer.echo("error: --model is required (see rukh eval --help)", err=True)
        raise typer.Exit(code=2)
    if suite not in SUITES:
        typer.echo(f"error: --suite must be one of {', '.join(SUITES)}", err=True)
        raise typer.Exit(code=2)
    if not Path(model).is_file() and not is_hub_id(model):
        typer.echo(
            f"error: --model {model!r} is neither an existing checkpoint nor a Hub id "
            "of the form owner/name",
            err=True,
        )
        raise typer.Exit(code=2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = load_suite(suite, config)
    if stage is not None:
        cfg = cfg.model_copy(update={"stage": stage})
    result, report = run_suite(model, cfg, suite=suite, use_cache=not no_cache, device=device)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"stage:    {result.stage} ({result.params:,} parameters)")
    typer.echo(f"suite:    {result.suite}{' (cache off)' if no_cache else ''} on {result.device}")
    if result.legality_argmax:
        typer.echo(f"legality: {result.legality_argmax.rate:.4f} argmax, unmasked")
    if result.legality_sampled:
        typer.echo(f"          {result.legality_sampled.rate:.4f} sampled, unmasked")
    if result.accuracy:
        typer.echo(f"accuracy: top1 {result.accuracy.top1:.4f}  top3 {result.accuracy.top3:.4f}")
    if result.puzzles:
        typer.echo(f"puzzles:  {result.puzzles.rate:.4f} solved")
    if result.elo:
        typer.echo(f"elo:      {elo_line(result.elo)} over {result.elo.games} games")
    for note in result.notes:
        typer.echo(f"note:     {note}")
    typer.echo(f"report:   {report.markdown}")
    typer.echo(f"results:  {report.results}")
    if report.web:
        typer.echo(f"table:    {report.web}")


@eval_app.command("drop")
def eval_drop_cmd(
    stages: Annotated[
        str, typer.Option("--stages", help="Rows to remove from the table, comma separated.")
    ],
    table: Annotated[
        Path | None, typer.Option("--table", help="Results file (default: the shared one).")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Do it without asking.")] = False,
) -> None:
    """Retract rows from the single results table.

    A measurement can turn out to be wrong, and when it does the table has to be able to say so by
    not carrying it any more. It happened once: D-070 found four of the eight rungs of the Elo
    ladder had invented ratings, wrong by about five hundred points. The corrected runs went in
    under new stage names, so the old rows stayed and the project page kept serving retracted
    numbers.
    """
    from rukh.eval.report import drop_rows
    from rukh.paths import resolve

    target = table or resolve("artifacts/web/results.json")
    wanted = [name.strip() for name in stages.split(",") if name.strip()]
    if not wanted:
        typer.echo("error: --stages must name at least one row", err=True)
        raise typer.Exit(code=2)
    if not yes and not typer.confirm(f"Remove {', '.join(wanted)} from {target}?"):
        raise typer.Abort
    removed, kept = drop_rows(target, wanted)
    missing = sorted(set(wanted) - set(removed))
    for name in removed:
        typer.echo(f"removed:  {name}")
    for name in missing:
        typer.echo(f"not there: {name}")
    typer.echo(f"rows left: {len(kept)}")


@eval_app.command("openings")
def eval_openings_cmd(
    model: Annotated[str, typer.Option("--model", help="Checkpoint or Hub id to read.")],
    against: Annotated[
        str | None,
        typer.Option("--against", help="A second model to compare the distribution with."),
    ] = None,
    header_elo: Annotated[
        int, typer.Option("--elo", help="Elo header to ask the distribution at.")
    ] = 1800,
    top: Annotated[int, typer.Option("--top", help="How many first moves to print.")] = 8,
    device: Annotated[str | None, typer.Option("--device", help="Where to run.")] = None,
) -> None:
    """Print what the model would open with, as probabilities rather than as sampled games.

    No temperature, no top-k, no seed: the softmax over the twenty legal first moves as the
    weights produce it. That is what makes it the right instrument for an adapter's effect --
    the share it moved is a property of the weights and does not have to be averaged out of
    hundreds of self-play games.
    """
    from rukh.eval.diversity import first_move_distribution, first_move_entropy
    from rukh.eval.suite import resolve_model
    from rukh.tokenize.uci_vocab import UciTokenizer
    from rukh.train import load_model, pick_device

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    where = device or pick_device()
    tok = UciTokenizer()

    def read(path: str) -> tuple[dict[str, float], float]:
        loaded, _ = load_model(resolve_model(path), map_location=where)
        loaded = loaded.to(where).eval()
        return first_move_distribution(loaded, tok, header_elo), first_move_entropy(
            loaded, tok, header_elo
        )

    try:
        probs, entropy = read(model)
        other = read(against) if against is not None else None
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"header:   <w{header_elo:04d}>")
    typer.echo(f"entropy:  {entropy:.4f} bits of {math.log2(20):.4f}")
    if other is not None:
        typer.echo(f"          {other[1]:.4f} bits for {against}")
    order = sorted(probs, key=lambda move: probs[move], reverse=True)[:top]
    header_row = "move      share" + ("     other     delta" if other is not None else "")
    typer.echo(header_row)
    for move in order:
        line = f"{move:<9s} {probs[move] * 100:6.2f} %"
        if other is not None:
            mine, theirs = probs[move], other[0].get(move, 0.0)
            line += f"  {theirs * 100:6.2f} %  {(theirs - mine) * 100:+6.2f}"
        typer.echo(line)


@eval_app.command("qwen")
def eval_qwen_cmd(
    adapter: Annotated[
        Path,
        typer.Option(
            "--adapter", exists=True, file_okay=False, help="Directory of the fine-tuned adapter."
        ),
    ],
    suite: Annotated[str, typer.Option("--suite", help="Suite name: full or quick.")] = "full",
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, readable=True, help="Suite YAML override."
        ),
    ] = None,
    stage: Annotated[str, typer.Option("--stage", help="Row name in the results table.")] = (
        "qwen3-pgn-qlora"
    ),
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Recompute everything.")] = False,
    device: Annotated[str | None, typer.Option("--device", help="Where to run.")] = None,
) -> None:
    """Put a fine-tuned general model through the decoder's own suite.

    Same positions, same puzzles, same Stockfish ladder. The one column this report has and the
    decoder's does not is the breakdown of *how* its answers failed: a model writing SAN can write
    something that is not a move, or a move two pieces could make, and neither is possible for a
    vocabulary in which one token is one move.
    """
    from rukh.eval import load_suite
    from rukh.eval.qwen_suite import run_qwen_suite
    from rukh.eval.report import elo_line
    from rukh.eval.suite import SUITES

    if suite not in SUITES:
        typer.echo(f"error: --suite must be one of {', '.join(SUITES)}", err=True)
        raise typer.Exit(code=2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = load_suite(suite, config)
    try:
        result, report = run_qwen_suite(
            adapter, cfg, stage=stage, use_cache=not no_cache, device=device
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    stats = result.written
    typer.echo(f"stage:    {result.stage} ({result.params:,} parameters)")
    typer.echo(f"legal:    {stats.legal_rate:.4f} of {stats.asked} answers")
    typer.echo(
        f"failures: {stats.illegal} illegal, {stats.unparseable} not a move, "
        f"{stats.ambiguous} ambiguous, {stats.empty} empty"
    )
    if result.top1 is not None:
        typer.echo(f"accuracy: top1 {result.top1:.4f}")
    if result.puzzles is not None:
        typer.echo(f"puzzles:  {result.puzzles.rate:.4f} solved")
    if result.elo is not None:
        typer.echo(f"elo:      {elo_line(result.elo)} over {result.elo.games} games")
    typer.echo(f"report:   {report.markdown}")


@eval_app.command("sweep")
def eval_sweep_cmd(
    model: Annotated[
        str, typer.Option("--model", help="Checkpoint path or Hub id to evaluate per condition.")
    ],
    elos: Annotated[
        str,
        typer.Option("--elos", help="Increasing Elo headers to ask for, comma separated."),
    ] = "1200,1500,1800,2100,2400",
    suite: Annotated[str, typer.Option("--suite", help="Suite name: full or quick.")] = "full",
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, readable=True, help="Suite YAML override."
        ),
    ] = None,
    stage: Annotated[str | None, typer.Option("--stage", help="Name of the sweep.")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Recompute every game.")] = False,
    device: Annotated[str | None, typer.Option("--device", help="Where to run.")] = None,
    web: Annotated[
        bool, typer.Option("--web/--no-web", help="Also write the JSON the course reads.")
    ] = True,
) -> None:
    """Run the suite once per Elo header and say whether the ratings come out monotonic.

    Every row is the same evaluation with one number changed, so anything that differs between
    them is the condition. The verdict is printed twice on purpose: whether the point estimates
    rise, and whether the confidence intervals actually separate. Only the second is evidence.
    """
    from rukh.eval import load_suite
    from rukh.eval.suite import SUITES, is_hub_id
    from rukh.eval.sweep import WEB_FILE, elo_summary, run_sweep, write_sweep
    from rukh.paths import resolve

    if suite not in SUITES:
        typer.echo(f"error: --suite must be one of {', '.join(SUITES)}", err=True)
        raise typer.Exit(code=2)
    if not Path(model).is_file() and not is_hub_id(model):
        typer.echo(f"error: --model {model!r} is neither a checkpoint nor a Hub id", err=True)
        raise typer.Exit(code=2)
    try:
        wanted = [int(part) for part in elos.split(",") if part.strip()]
    except ValueError as exc:
        typer.echo(f"error: --elos must be a comma-separated list of integers: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = load_suite(suite, config)
    try:
        result = run_sweep(
            model, cfg, wanted, suite=suite, use_cache=not no_cache, device=device, stage=stage
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    markdown = write_sweep(result, resolve(cfg.out_dir), WEB_FILE if web else None)
    typer.echo(f"stage:      {result.stage} ({result.params:,} parameters)")
    typer.echo(elo_summary(result.rows))
    typer.echo(f"monotonic:  {'yes' if result.monotonic else 'no'} (point estimates)")
    typer.echo(f"separated:  {'yes' if result.separated else 'no'} (confidence intervals)")
    if result.span is not None:
        typer.echo(f"span:       {result.span:.0f} Elo")
    typer.echo(f"report:     {markdown}")


@eval_app.command("encoder")
def eval_encoder_cmd(
    model: Annotated[
        Path,
        typer.Option(
            "--model", exists=True, dir_okay=False, readable=True, help="Fine-tuned checkpoint."
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, readable=True, help="Suite YAML override."
        ),
    ] = None,
    stage: Annotated[
        str | None, typer.Option("--stage", help="Row name in the results table.")
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Where to run: cuda, cpu... (default: the training device)."),
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Recompute every baseline verdict.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON only.")] = False,
) -> None:
    """Measure the encoder's heads against the labels and the material baseline."""
    from rukh.eval.encoder import load_encoder_suite, run_encoder_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = load_encoder_suite(config)
    if stage is not None:
        cfg = cfg.model_copy(update={"stage": stage})
    try:
        result, report = run_encoder_suite(model, cfg, use_cache=not no_cache, device=device)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"stage:    {result.stage} ({result.params:,} parameters)")
    typer.echo(
        f"items:    {result.items} positions, {result.blunder_items} with a blunder label "
        f"on {result.device}"
    )
    typer.echo(
        f"split:    tune {result.tune_items} rows / {result.tune_games} games, "
        f"score {result.score_items} rows / {result.score_games} games "
        f"(base rate {(result.blunder_base_rate or 0.0) * 100:.2f} %)"
    )
    tuned = "n/a" if result.threshold_tuned is None else f"{result.threshold_tuned:.4g}"
    fixed = "n/a" if result.threshold_fixed is None else f"{result.threshold_fixed:.4g}"
    for measured, point in (
        (result.encoder_blunder, f"p>={tuned} tuned"),
        (result.encoder_blunder_fixed, f"p>={fixed} fixed"),
        (result.heuristic_blunder, "rule, untuned"),
    ):
        if measured is not None:
            typer.echo(
                f"blunder:  {measured.name:<15} {point:<16} P {measured.precision:.4f}  "
                f"R {measured.recall:.4f}  F1 {measured.f1:.4f}"
            )
    if result.encoder_blunder_ranking is not None:
        rank = result.encoder_blunder_ranking
        auc = "n/a" if rank.roc_auc is None else f"{rank.roc_auc:.4f}"
        average = "n/a" if rank.average_precision is None else f"{rank.average_precision:.4f}"
        typer.echo(
            f"ranking:  roc auc {auc}  average precision {average}  "
            f"(a coin flip scores 0.5000 and {rank.base_rate:.4f})"
        )
    if result.f1_margin is not None:
        verdict = "meets the bar" if result.meets_goal else "below the bar"
        typer.echo(f"margin:   {result.f1_margin:+.1f} F1 points over the baseline ({verdict})")
    if result.encoder_value is not None:
        from rukh.eval.encoder import GOAL_VALUE_CORRELATION

        met = result.value_correlation_meets_goal
        verdict = "not measured" if met is None else ("meets the bar" if met else "below the bar")
        typer.echo(
            f"value:    vs {result.encoder_value.target}  "
            f"spearman {result.encoder_value.spearman}  "
            f"pearson {result.encoder_value.pearson}  "
            f"(>= {GOAL_VALUE_CORRELATION:.2f}: {verdict})"
        )
    if result.result_accuracy is not None:
        typer.echo(f"result:   {result.result_accuracy:.4f} accuracy")
    for point in result.label_curve:
        typer.echo(f"curve:    {point.fraction:>5.0%} of the labels ({point.train_labels} rows)")
    for note in result.notes:
        typer.echo(f"note:     {note}")
    typer.echo(f"report:   {report.markdown}")
    typer.echo(f"results:  {report.results}")
    if report.web:
        typer.echo(f"table:    {report.web}")


@app.command("export")
def export_cmd(
    ckpt: Annotated[
        Path,
        typer.Option(
            "--ckpt", exists=True, dir_okay=False, readable=True, help="Checkpoint to export."
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory (or .onnx file).")],
    kind: Annotated[
        str, typer.Option("--kind", help="What to export: decoder or encoder.")
    ] = "decoder",
    opset: Annotated[int, typer.Option("--opset", help="ONNX opset version.")] = 18,
    seq_len: Annotated[
        int, typer.Option("--seq-len", help="Example length the exporter traces.")
    ] = 200,
    fp16: Annotated[bool, typer.Option("--fp16", help="Also write the fp16 model.")] = False,
    int8: Annotated[bool, typer.Option("--int8", help="Also write the int8 model.")] = False,
    check_parity: Annotated[
        bool, typer.Option("--check-parity", help="Compare every file with PyTorch.")
    ] = False,
    positions: Annotated[
        int, typer.Option("--positions", help="Positions used by the parity check.")
    ] = 1000,
    games: Annotated[
        Path | None,
        typer.Option(
            "--games",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Validation games parquet for the parity positions.",
        ),
    ] = None,
    adapter_inputs: Annotated[
        bool,
        typer.Option(
            "--adapter-inputs",
            help="Write the graph with the LoRA factors as inputs, so styles can be swapped.",
        ),
    ] = False,
    adapter: Annotated[
        Path | None,
        typer.Option(
            "--adapter",
            exists=True,
            file_okay=False,
            readable=True,
            help="Adapter folder to check the parity of the swapped path against PyTorch.",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON only.")] = False,
) -> None:
    """Export a model to ONNX, quantize it and check parity with PyTorch.

    The decoder exports its next-move head; ``--kind encoder`` exports the two heads the demo
    reads from a position, ``value`` and ``blunder``. With ``--adapter-inputs`` the decoder's
    graph takes its LoRA factors as two extra inputs: fed zeros it is the checkpoint, fed a
    1.6 MB adapter it is that style, and the browser changes style without downloading a model.
    """
    from rukh.export import KINDS, export_all

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    if kind not in KINDS:
        typer.echo(f"error: --kind must be one of {', '.join(KINDS)}", err=True)
        raise typer.Exit(code=2)
    try:
        bundle = export_all(
            ckpt,
            out,
            kind=kind,
            opset=opset,
            seq_len=seq_len,
            fp16=fp16,
            int8=int8,
            check_parity=check_parity,
            positions=positions,
            games=games,
            adapter_inputs=adapter_inputs,
            adapter=adapter,
        )
    except (ImportError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(bundle.model_dump_json(indent=2))
        return
    typer.echo(f"exporter: {bundle.onnx.exporter} (opset {bundle.onnx.opset})")
    typer.echo(f"kind:     {bundle.onnx.kind} -> {', '.join(bundle.onnx.outputs)}")
    if bundle.onnx.warning:
        typer.echo(f"warning:  {bundle.onnx.warning}")
    checked = "verified" if bundle.onnx.dynamic_seq_verified else "not verified"
    typer.echo(
        f"shapes:   batch dynamic={bundle.onnx.dynamic_batch}, "
        f"sequence dynamic={bundle.onnx.dynamic_seq} ({checked}), block={bundle.onnx.block}"
    )
    typer.echo(f"fp32:     {bundle.onnx.path} ({bundle.onnx.bytes} bytes)")
    for quantized in (bundle.fp16, bundle.int8):
        if quantized is not None:
            typer.echo(
                f"{quantized.kind}:     {quantized.path} ({quantized.bytes} bytes, "
                f"{quantized.ratio:.2f} of fp32, {quantized.method})"
            )
    if bundle.adapter_inputs:
        shape_a = bundle.onnx.metadata.get("rukh_adapter_shape_a", "?")
        shape_b = bundle.onnx.metadata.get("rukh_adapter_shape_b", "?")
        typer.echo(f"adapter:  factors are inputs, A ({shape_a}) and B ({shape_b})")
    for name, result in bundle.parity.items():
        label = "parity" if not bundle.adapter_inputs else "parity zeros"
        typer.echo(
            f"{label} {name}: {result.agreement:.4f} on {result.positions} "
            f"{bundle.parity_source} positions "
            f"(max |delta logits| {result.max_abs_logit_delta:.4g})"
        )
    for name, result in bundle.adapter_parity.items():
        typer.echo(
            f"parity {bundle.adapter_name} {name}: {result.agreement:.4f} on "
            f"{result.positions} positions "
            f"(max |delta logits| {result.max_abs_logit_delta:.4g})"
        )
    for name, heads in bundle.heads_parity.items():
        typer.echo(
            f"parity {name}: {heads.agreement:.4f} of the blunder decisions on "
            f"{heads.positions} {bundle.parity_source} positions "
            f"(max |delta value| {heads.max_abs_value_delta:.4g})"
        )
    if bundle.parity_path:
        typer.echo(f"parity:   {bundle.parity_path}")
    if bundle.parity_warning:
        typer.echo(f"warning:  {bundle.parity_warning}")


@encoder_app.command("embed")
def encoder_embed_cmd(
    positions: Annotated[
        Path,
        typer.Option(
            "--positions",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Parquet with a `fen` column (the P1 positions or evaluations table).",
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="Where to write the .npy array.")],
    ckpt: Annotated[
        Path,
        typer.Option(
            "--ckpt", exists=True, dir_okay=False, readable=True, help="Encoder checkpoint."
        ),
    ],
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Positions per forward pass.")
    ] = 256,
    device: Annotated[
        str | None, typer.Option("--device", help="Where to run: cuda, cpu... ")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON only.")] = False,
) -> None:
    """Write mean-pooled position embeddings plus the `fen4` sidecar that names their rows."""
    from rukh.export import embed_positions

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    try:
        result = embed_positions(ckpt, positions, out, batch_size=batch_size, device=device)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"positions: {result.positions}")
    typer.echo(f"dim:       {result.dim} ({result.pooling} pooling, {result.input} scheme)")
    typer.echo(f"array:     {result.array}")
    typer.echo(f"sidecar:   {result.sidecar} (row, fen4)")


@publish_app.command("model")
def publish_model_cmd(
    ckpt: Annotated[
        Path,
        typer.Option(
            "--ckpt", exists=True, dir_okay=False, readable=True, help="Checkpoint to publish."
        ),
    ],
    repo: Annotated[str, typer.Option("--repo", help="Hub repository, e.g. chorcat/rukh-small.")],
    onnx: Annotated[
        Path | None,
        typer.Option("--onnx", exists=True, file_okay=False, help="Directory with the ONNX files."),
    ] = None,
    stage: Annotated[
        str | None,
        typer.Option("--stage", help="Stage name; default: the repo without `rukh-`."),
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="MLflow run to take the recipe from.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Stage the folder locally; touch no network.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON only.")] = False,
) -> None:
    """Stage weights, ONNX, tokenizer and a generated card, and upload them to the Hub."""
    from rukh.publish import ModelPublishConfig, publish_model

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    try:
        result = publish_model(
            ckpt,
            repo,
            ModelPublishConfig(),
            onnx_dir=onnx,
            stage=stage,
            run_id=run_id,
            dry_run=dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"repo:     {result.repo_id} ({result.repo_type}, {result.kind})")
    typer.echo(f"stage:    {result.stage} ({result.params:,} parameters)")
    typer.echo(f"mode:     {'dry-run (staged, nothing uploaded)' if dry_run else 'uploaded'}")
    typer.echo(f"weights:  {result.weights_format}")
    typer.echo(f"run:      {result.run_id or 'not found in MLflow'}")
    typer.echo(f"folder:   {result.folder}")
    typer.echo("files:")
    for path in result.files:
        typer.echo(f"  {path}")


@publish_app.command("adapter")
def publish_adapter_cmd(
    run_dir: Annotated[
        Path,
        typer.Option(
            "--run", exists=True, file_okay=False, help="Training run folder with the adapter."
        ),
    ],
    repo: Annotated[str, typer.Option("--repo", help="Hub repository, e.g. chorcat/rukh-lora-e4.")],
    base: Annotated[
        str, typer.Option("--base", help="Repository of the model this adapter mounts on.")
    ],
    stage: Annotated[str | None, typer.Option("--stage", help="Name of the adapter.")] = None,
    effect: Annotated[
        Path | None,
        typer.Option(
            "--effect",
            exists=True,
            dir_okay=False,
            help="JSON with what the adapter changed (AdapterEffect).",
        ),
    ] = None,
    n_layer: Annotated[int, typer.Option("--n-layer", help="Layers of the base model.")] = 16,
    d_model: Annotated[int, typer.Option("--d-model", help="Width of the base model.")] = 768,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Stage the folder locally; touch no network.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON only.")] = False,
) -> None:
    """Publish a LoRA adapter as its own repository, with the base model it needs to work.

    An adapter is not a model: no weights, no ONNX, no vocabulary, and on its own it does nothing
    at all. So the card leads with the base model, and the numbers it publishes are what the
    adapter *changed* -- publishing it with the base model's Elo would be publishing somebody
    else's number.
    """
    from rukh.publish.adapter import AdapterEffect, publish_adapter

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    measured = (
        AdapterEffect.model_validate_json(effect.read_text(encoding="utf-8"))
        if effect is not None
        else None
    )
    try:
        result = publish_adapter(
            run_dir,
            repo,
            base,
            stage=stage,
            effect=measured,
            base_config={"n_layer": n_layer, "d_model": d_model},
            dry_run=dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"repo:     {result.repo_id}")
    typer.echo(f"base:     {result.base_repo}")
    typer.echo(f"mode:     {'dry-run (staged, nothing uploaded)' if dry_run else 'uploaded'}")
    typer.echo(f"params:   {result.params:,} ({result.bytes / 1e6:.1f} MB)")
    typer.echo(f"folder:   {result.folder}")
    for path in result.files:
        typer.echo(f"  {path}")


@publish_app.command("qwen")
def publish_qwen_cmd(
    run_dir: Annotated[
        Path,
        typer.Option("--run", exists=True, file_okay=False, help="The peft output folder."),
    ],
    repo: Annotated[str, typer.Option("--repo", help="Hub repository for the adapter.")],
    stage: Annotated[
        str, typer.Option("--stage", help="Evaluation row the card reads.")
    ] = "qwen3-pgn-qlora",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Write the card locally; touch no network.")
    ] = False,
) -> None:
    """Publish the QLoRA adapter of the general model, in the format `peft` wrote it.

    Not re-staged into this project's own adapter format: `peft` already wrote a valid
    `adapter_config.json` and `adapter_model.safetensors`, and rewriting them would make the file
    unusable with the two lines of `peft` any reader would actually type. What gets added is the
    card, built from the run's own record and from the evaluation, never by hand.
    """
    from rukh.publish.adapter import publish_qwen_adapter

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    try:
        result = publish_qwen_adapter(run_dir, repo, stage=stage, dry_run=dry_run)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"repo:     {result.repo_id}")
    typer.echo(f"base:     {result.base_repo}")
    typer.echo(f"mode:     {'dry-run (card written, nothing uploaded)' if dry_run else 'uploaded'}")
    typer.echo(f"trained:  {result.params:,} parameters ({result.bytes / 1e6:.1f} MB)")
    typer.echo(f"card:     {result.card_path}")
    for path in result.files:
        typer.echo(f"  {path}")


@mlflow_app.command("ui")
def mlflow_ui(
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 5000,
) -> None:
    """Open the MLflow UI on the local SQLite store (blocks until stopped)."""
    from rukh.tracking import serve_ui, tracking_uri

    typer.echo(f"mlflow ui on http://{host}:{port} (store {tracking_uri()})")
    raise typer.Exit(code=serve_ui(host=host, port=port))


@engine_app.command("check")
def engine_check_cmd(
    elo: Annotated[int, typer.Option("--elo", help="UCI_Elo to configure.")] = 1400,
    plies: Annotated[int, typer.Option("--plies", help="Half-moves to play.")] = 40,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the random opponent.")] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON only.")] = False,
) -> None:
    """Locate Stockfish, set UCI_Elo and play a short game against a random mover."""
    from rukh.engine import EngineCheckResult, EngineError, EngineNotFound, engine_check

    try:
        result = engine_check(elo=elo, plies=plies, seed=seed)
    except (EngineNotFound, EngineError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    width = max(len(key) for key in EngineCheckResult.model_fields)
    for key, value in result.model_dump().items():
        typer.echo(f"{key:<{width}}  {value if value is not None else 'null'}")
