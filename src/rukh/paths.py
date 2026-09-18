"""Project directories.

Every accessor reads the environment at call time so tests (and users) can point the whole
project elsewhere with ``RUKH_HOME`` without re-importing the package.
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def root() -> Path:
    """Repository root: ``RUKH_HOME`` if set, otherwise the parent of ``src/``."""
    home = os.environ.get("RUKH_HOME")
    return Path(home).expanduser().resolve() if home else _PACKAGE_ROOT


def data_dir() -> Path:
    """Where raw and derived datasets live (gitignored)."""
    return root() / "data"


def mlruns_dir() -> Path:
    """Where the local MLflow store lives (gitignored)."""
    return root() / "mlruns"


def tools_dir() -> Path:
    """Where downloaded binaries such as Stockfish live (gitignored)."""
    return root() / "tools"
