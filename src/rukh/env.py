"""Environment report: what Python, torch, GPU and tools this machine has."""

from __future__ import annotations

import platform

from pydantic import BaseModel, ConfigDict

from rukh import paths
from rukh.engine import find_stockfish
from rukh.tracking import tracking_uri


class EnvReport(BaseModel):
    """Snapshot of the runtime environment as seen by ``rukh info``."""

    model_config = ConfigDict(extra="forbid")

    python: str
    platform: str
    torch: str | None
    cuda: bool
    gpu: str | None
    stockfish: str | None
    mlflow_uri: str
    data_dir: str
    mlruns_dir: str


def _torch_info() -> tuple[str | None, bool, str | None]:
    """Return (torch version, cuda available, gpu name); all empty if torch is missing."""
    try:
        import torch
    except ImportError:
        return None, False, None
    cuda = bool(torch.cuda.is_available())
    gpu = torch.cuda.get_device_name(0) if cuda else None
    return torch.__version__, cuda, gpu


def collect() -> EnvReport:
    """Collect the environment report. Never raises because torch or Stockfish are absent."""
    torch_version, cuda, gpu = _torch_info()
    stockfish = find_stockfish()
    return EnvReport(
        python=platform.python_version(),
        platform=f"{platform.system()}-{platform.release()}-{platform.machine()}",
        torch=torch_version,
        cuda=cuda,
        gpu=gpu,
        stockfish=str(stockfish) if stockfish else None,
        mlflow_uri=tracking_uri(create=False),
        data_dir=str(paths.data_dir()),
        mlruns_dir=str(paths.mlruns_dir()),
    )
