"""``PipelineConfig``: every data step's config under one YAML (``configs/data/pipeline.yaml``)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from rukh import paths
from rukh.config import BaseConfig, load_yaml
from rukh.data.db import DuckDbConfig
from rukh.data.elite import EliteConfig
from rukh.data.elo_bins import EloBinsConfig
from rukh.data.evals import EvalsConfig
from rukh.data.pairs import PairsConfig
from rukh.data.positions import PositionsConfig
from rukh.data.publish import PublishConfig
from rukh.data.puzzles import PuzzlesConfig
from rukh.data.uci import UciConfig


class TokenizeConfig(BaseConfig):
    """Where games come from, where packed tokens and tokenizer artifacts go."""

    uci_dir: str = "data/uci"
    out_dir: str = "data/tokens"
    artifacts_dir: str = "artifacts/tokenizer"
    web_artifacts_dir: str = "artifacts/web"
    fixture_pgn: str = "tests/fixtures/games.pgn"
    train_months: list[str] = Field(default_factory=lambda: ["2025-01"])
    val_month: str = "2025-02"
    val_games: int = Field(default=0, ge=0)
    extra_train_parquets: list[str] = Field(default_factory=list)
    """UCI parquets appended to the training split, outside the ``year=/month=`` layout."""
    val_remainder_trains: bool = True
    """Whether the part of ``val_month`` that is not held out joins the training split.

    Off for a corpus whose shape is the point: the Elo-balanced sample of M4 has the same number
    of games per rating band on purpose, and 2.85 M games of rated-1800+ play poured on top would
    undo it."""
    """Games of ``val_month`` held out; the rest of it trains. 0 = the whole month validates."""
    stats_games: str = "data/uci/year=2025/month=01/games.parquet"
    bpe_vocab_size: int = Field(default=4096, ge=100)
    bpe_train_games: int = Field(default=200_000, ge=1)
    stats_n_games: int = Field(default=20_000, ge=1)
    max_len: int = Field(default=200, ge=8)


DUCKDB_STEPS = ("positions", "evals", "puzzles", "elo_bins")


class PipelineConfig(BaseConfig):
    """All step configs; each step reads only its own section.

    ``duckdb:`` is shared: the top-level settings are copied into every step that opens a
    DuckDB connection unless that step declares its own.
    """

    duckdb: DuckDbConfig = Field(default_factory=DuckDbConfig)
    uci: UciConfig = Field(default_factory=UciConfig)
    tokenize: TokenizeConfig = Field(default_factory=TokenizeConfig)
    positions: PositionsConfig = Field(default_factory=PositionsConfig)
    evals: EvalsConfig = Field(default_factory=EvalsConfig)
    puzzles: PuzzlesConfig = Field(default_factory=PuzzlesConfig)
    pairs: PairsConfig = Field(default_factory=PairsConfig)
    elite: EliteConfig = Field(default_factory=EliteConfig)
    elo_bins: EloBinsConfig = Field(default_factory=EloBinsConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)

    @model_validator(mode="after")
    def _share_duckdb(self) -> PipelineConfig:
        default = DuckDbConfig()
        for name in DUCKDB_STEPS:
            step = getattr(self, name)
            if step.duckdb == default:
                setattr(self, name, step.model_copy(update={"duckdb": self.duckdb}))
        return self


def default_config_path() -> Path:
    return paths.package_root() / "configs" / "data" / "pipeline.yaml"


def load_pipeline(path: Path | None = None) -> PipelineConfig:
    """Load ``pipeline.yaml`` (the shipped one when ``path`` is ``None``)."""
    return load_yaml(path or default_config_path(), PipelineConfig)
