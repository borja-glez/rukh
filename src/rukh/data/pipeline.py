"""``PipelineConfig``: every data step's config under one YAML (``configs/data/pipeline.yaml``)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from rukh import paths
from rukh.config import BaseConfig, load_yaml


class TokenizeConfig(BaseConfig):
    """Where games come from, where packed tokens and tokenizer artifacts go."""

    uci_dir: str = "data/uci"
    out_dir: str = "data/tokens"
    artifacts_dir: str = "artifacts/tokenizer"
    web_artifacts_dir: str = "artifacts/web"
    fixture_pgn: str = "tests/fixtures/games.pgn"
    train_month: str = "2025-01"
    val_month: str = "2025-02"
    stats_games: str = "data/uci/year=2025/month=01/games.parquet"
    bpe_vocab_size: int = Field(default=4096, ge=100)
    bpe_train_games: int = Field(default=200_000, ge=1)
    stats_n_games: int = Field(default=20_000, ge=1)
    max_len: int = Field(default=200, ge=8)


class PipelineConfig(BaseConfig):
    """All step configs; each step reads only its own section."""

    tokenize: TokenizeConfig = Field(default_factory=TokenizeConfig)


def default_config_path() -> Path:
    return paths.package_root() / "configs" / "data" / "pipeline.yaml"


def load_pipeline(path: Path | None = None) -> PipelineConfig:
    """Load ``pipeline.yaml`` (the shipped one when ``path`` is ``None``)."""
    return load_yaml(path or default_config_path(), PipelineConfig)


def resolve(rel: str) -> Path:
    """Absolute path for a config entry: as is if absolute, else under ``paths.root()``."""
    path = Path(rel)
    return path if path.is_absolute() else paths.root() / path
