"""Local MLflow tracking: SQLite store under ``mlruns/`` and a run helper."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from rukh import __version__, paths

EXPERIMENT = "rukh"


def tracking_uri(create: bool = True) -> str:
    """SQLite tracking URI ``sqlite:///<MLRUNS_DIR>/mlflow.db``; creates the directory if asked."""
    mlruns = paths.mlruns_dir()
    if create:
        mlruns.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(mlruns / 'mlflow.db').as_posix()}"


def flatten(config: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested mappings into ``a.b=value`` string params (lists become their repr)."""
    flat: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = str(value)
    return flat


def git_sha() -> str | None:
    """Short SHA of the repository at ``paths.root()``, or None when there is no git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha or None


@contextmanager
def start_run(
    name: str,
    config: Mapping[str, Any],
    tags: Mapping[str, str] | None = None,
) -> Iterator[Any]:
    """Start an MLflow run in the local store with ``config`` logged as flattened params.

    Tags always include ``rukh_version`` and, when the project lives in a git repo, ``git_sha``.
    Yields the active ``mlflow.ActiveRun``.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT)
    run_tags: dict[str, str] = {"rukh_version": __version__}
    sha = git_sha()
    if sha:
        run_tags["git_sha"] = sha
    if tags:
        run_tags.update(tags)
    with mlflow.start_run(run_name=name, tags=run_tags) as run:
        params = flatten(config)
        if params:
            mlflow.log_params(params)
        yield run


def ui_command(host: str = "127.0.0.1", port: int = 5000) -> list[str]:
    """Argv for ``mlflow ui`` against the local store."""
    return [
        "mlflow",
        "ui",
        "--backend-store-uri",
        tracking_uri(),
        "--host",
        host,
        "--port",
        str(port),
    ]


def serve_ui(host: str = "127.0.0.1", port: int = 5000) -> int:
    """Run ``mlflow ui`` in the foreground and return its exit code."""
    return subprocess.call(ui_command(host=host, port=port))
