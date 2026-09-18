"""Orchestration behind ``rukh data tokenize``: tokenizer artifacts and the parity fixture."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rukh.data.pipeline import TokenizeConfig, resolve
from rukh.tokenize.fixture import build_fixture, write_fixture
from rukh.tokenize.uci_vocab import UciTokenizer

SCHEMES = ("uci", "san", "bpe")


class TokenizeReport(BaseModel):
    """What ``run`` wrote, as absolute POSIX paths."""

    model_config = ConfigDict(extra="forbid")

    scheme: str
    vocab_size: int
    written: list[str]


def export_artifacts(cfg: TokenizeConfig) -> list[Path]:
    """Write ``vocab.json`` and ``fixtures/games.json`` under ``artifacts_dir``."""
    artifacts = resolve(cfg.artifacts_dir)
    tokenizer = UciTokenizer()
    vocab_path = artifacts / "vocab.json"
    tokenizer.export(vocab_path)
    entries = build_fixture(resolve(cfg.fixture_pgn), tokenizer, max_len=cfg.max_len)
    fixture_path = artifacts / "fixtures" / "games.json"
    write_fixture(entries, fixture_path)
    return [vocab_path, fixture_path]


def run(cfg: TokenizeConfig, scheme: str = "uci", export_fixture: bool = False) -> TokenizeReport:
    """Run the requested tokenizer step(s) for ``scheme``."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    written: list[Path] = []
    if export_fixture:
        written.extend(export_artifacts(cfg))
    return TokenizeReport(
        scheme=scheme, vocab_size=len(UciTokenizer()), written=[p.as_posix() for p in written]
    )
