"""Command-line interface: ``info``, ``data ...``, ``engine check`` and ``mlflow ui``."""

from __future__ import annotations

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
    as_json: Annotated[bool, typer.Option("--json", help="Print the plan as JSON only.")] = False,
) -> None:
    """Fetch filtered Lichess games as parquet files plus a manifest."""
    from rukh.config import load_yaml
    from rukh.data.fetch import FetchConfig, run

    cfg = load_yaml(config, FetchConfig)
    fetch_plan = run(cfg, dry_run=dry_run)
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
) -> None:
    """Build tokenizer artifacts, the Python/TypeScript parity fixture and statistics."""
    from rukh.data.pipeline import load_pipeline
    from rukh.tokenize.run import SCHEMES, run

    if scheme not in SCHEMES:
        typer.echo(f"error: --scheme must be one of {', '.join(SCHEMES)}", err=True)
        raise typer.Exit(code=2)
    cfg = load_pipeline(config).tokenize
    report = run(cfg, scheme=scheme, export_fixture=export_fixture, stats=stats, games=games)
    typer.echo(f"scheme:     {report.scheme}")
    typer.echo(f"vocab_size: {report.vocab_size}")
    _echo_written(report.written)


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
