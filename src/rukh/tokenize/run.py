"""Orchestration behind ``rukh data tokenize``: artifacts, parity fixture, statistics, packing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from rukh.data.pipeline import TokenizeConfig
from rukh.paths import resolve
from rukh.tokenize.fixture import build_fixture, write_fixture
from rukh.tokenize.san_chars import SanCharTokenizer, san_text, uci_to_san
from rukh.tokenize.uci_vocab import UciTokenizer

if TYPE_CHECKING:
    from tokenizers import Tokenizer

SCHEMES = ("uci", "san", "bpe")


class TokenizeReport(BaseModel):
    """What ``run`` wrote, as absolute POSIX paths."""

    model_config = ConfigDict(extra="forbid")

    scheme: str
    vocab_size: int
    written: list[str]


def _bpe_path(cfg: TokenizeConfig) -> Path:
    return resolve(cfg.artifacts_dir) / "bpe.json"


def _ensure_bpe(cfg: TokenizeConfig, games: Path, retrain: bool) -> Tokenizer:
    """Load ``bpe.json``; train it from ``games`` when missing or ``retrain``."""
    from rukh.tokenize.bpe import load_bpe, train_bpe_from_parquet

    path = _bpe_path(cfg)
    if retrain or not path.is_file():
        return train_bpe_from_parquet(games, path, cfg.bpe_vocab_size, cfg.bpe_train_games)
    return load_bpe(path)


def export_artifacts(cfg: TokenizeConfig, bpe: Tokenizer | None) -> list[Path]:
    """Write ``vocab.json`` and ``fixtures/games.json`` (with ``bpe_ids`` when ``bpe`` given)."""
    artifacts = resolve(cfg.artifacts_dir)
    tokenizer = UciTokenizer()
    vocab_path = artifacts / "vocab.json"
    tokenizer.export(vocab_path)
    entries = build_fixture(
        resolve(cfg.fixture_pgn), tokenizer, SanCharTokenizer(), bpe, max_len=cfg.max_len
    )
    fixture_path = artifacts / "fixtures" / "games.json"
    write_fixture(entries, fixture_path)
    return [vocab_path, fixture_path]


def _summary(lengths: list[int], max_len: int) -> dict[str, float]:
    arr = np.asarray(lengths, dtype=np.int64)
    return {
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "pct_le_max_len": round(float((arr <= max_len).mean() * 100), 2),
    }


def compute_stats(cfg: TokenizeConfig, games: Path, bpe: Tokenizer) -> dict[str, object]:
    """Tokens per game for the three schemes over the first ``stats_n_games`` of ``games``.

    Every count is what the scheme's encoder really emits (``rukh.tokenize.pack``): the UCI
    scheme frames a game with five tokens (``<bos>``, two Elo tokens, result, ``<eos>``), BPE
    with three (``<bos>``, result, ``<eos>``) and char-level SAN with two, because there the
    result is part of the text.
    """
    import polars as pl

    from rukh.tokenize.bpe import bpe_text, longest_tokens

    frame = (
        pl.scan_parquet(Path(games).as_posix())
        .select("uci", "white_elo", "black_elo", "result")
        .head(cfg.stats_n_games)
        .collect()
    )
    uci_tok = UciTokenizer()
    san_tok = SanCharTokenizer()
    uci_len: list[int] = []
    san_len: list[int] = []
    bpe_len: list[int] = []
    for uci, w_elo, b_elo, result in frame.iter_rows():
        uci_len.append(len(uci_tok.encode_game(uci, w_elo, b_elo, result, max_len=1 << 30)))
        san_len.append(len(san_tok.encode(san_text(uci_to_san(uci), result))))
        bpe_len.append(len(bpe.encode(bpe_text(uci)).ids) + 3)
    return {
        "source": {"games": Path(games).name, "n_games": frame.height, "max_len": cfg.max_len},
        "schemes": {
            "uci": {"vocab_size": len(uci_tok), **_summary(uci_len, cfg.max_len)},
            "san": {"vocab_size": len(san_tok), **_summary(san_len, cfg.max_len)},
            "bpe": {"vocab_size": bpe.get_vocab_size(), **_summary(bpe_len, cfg.max_len)},
        },
        "bpe_longest_tokens": longest_tokens(bpe, 10),
    }


def write_stats(cfg: TokenizeConfig, games: Path, bpe: Tokenizer) -> Path:
    path = resolve(cfg.web_artifacts_dir) / "tokenizer-stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = compute_stats(cfg, games, bpe)
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def run(
    cfg: TokenizeConfig,
    scheme: str = "uci",
    export_fixture: bool = False,
    stats: bool = False,
    games: Path | None = None,
    pack: bool = False,
) -> TokenizeReport:
    """Run the requested tokenizer step(s) for ``scheme``.

    ``--scheme bpe`` (re)trains ``bpe.json`` from ``games``; ``--stats`` needs a BPE, so it
    trains one from ``games`` when none exists. ``--export-fixture`` adds ``bpe_ids`` to the
    fixture whenever ``bpe.json`` is available. ``--pack`` writes the ``train`` and
    ``val`` splits as memmap token streams under ``out_dir/<scheme>/``.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    games_path = games or resolve(cfg.stats_games)
    written: list[Path] = []
    bpe: Tokenizer | None = None

    if scheme == "bpe" or stats:
        bpe = _ensure_bpe(cfg, games_path, retrain=scheme == "bpe")
        if scheme == "bpe":
            written.append(_bpe_path(cfg))
    elif (export_fixture or pack) and _bpe_path(cfg).is_file():
        from rukh.tokenize.bpe import load_bpe

        bpe = load_bpe(_bpe_path(cfg))

    if stats and bpe is not None:
        written.append(write_stats(cfg, games_path, bpe))
    if pack:
        from rukh.tokenize.pack import pack_scheme

        written.extend(pack_scheme(cfg, scheme, bpe))
    if export_fixture:
        written.extend(export_artifacts(cfg, bpe))

    vocab_size = len(UciTokenizer())
    if scheme == "san":
        vocab_size = len(SanCharTokenizer())
    elif scheme == "bpe" and bpe is not None:
        vocab_size = bpe.get_vocab_size()
    return TokenizeReport(
        scheme=scheme, vocab_size=vocab_size, written=[p.as_posix() for p in written]
    )
