"""Command-line interface: ``rukh info``, ``rukh data fetch``."""

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
data_app = typer.Typer(help="Datasets: fetch and describe Lichess games.", no_args_is_help=True)
app.add_typer(data_app, name="data")


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
    typer.echo("outputs:")
    for path in fetch_plan.out_paths:
        typer.echo(f"  {path}")
    typer.echo(f"manifest: {fetch_plan.manifest_path}")
