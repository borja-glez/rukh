"""Parity between PyTorch and ONNX Runtime: the same move, and how far the logits drifted.

The acceptance criterion of ``docs/spec/02`` is the *move*, not the logits: the demo picks an
``argmax`` (or samples from a truncated distribution), so a file that disagrees with PyTorch on
0.1 % of positions is fine and one that agrees on the numbers but not on the move is not. Both
are reported: ``agreement`` is the share of positions where the chosen token matches, and
``max_abs_logit_delta`` is the worst absolute difference seen anywhere in the logits, which is
what tells fp32, fp16 and int8 apart.

The positions are the thousand **validation** prefixes of ``docs/spec/02`` §6, the same ones the
legality and accuracy metrics use: a random legal walk visits positions no human would reach, so
agreeing on them says little about the file the demo will load. ``random_prefixes`` stays for the
case where no validation parquet is around (and for the tests), and which of the two was used is
reported alongside the number.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from pathlib import Path

import chess
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from rukh.export.onnx import INPUT_NAME, LastStepLogits
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

log = logging.getLogger(__name__)

DEFAULT_N = 1_000
MAX_MISMATCHES = 20
DEFAULT_GAMES = "data/uci/year=2025/month=02/games.parquet"
"""The validation month of P1: the parity positions come from here when it exists."""


class ParityResult(BaseModel):
    """How well an exported file reproduces the checkpoint."""

    model_config = ConfigDict(extra="forbid")

    positions: int
    agreement: float
    max_abs_logit_delta: float
    mismatches: list[int]
    """Indices of the first few positions where the chosen token differed."""


def random_prefixes(
    tok: UciTokenizer,
    n: int,
    seed: int = 0,
    min_ply: int = 1,
    max_ply: int = 40,
) -> list[list[int]]:
    """Token-id prefixes of random legal games, for a parity check without any dataset.

    Every prefix is a real sequence of legal moves, so the model is asked the kind of question
    the demo asks it, and the walk is seeded so a parity number can be reproduced exactly.
    """
    rng = random.Random(seed)
    prefixes: list[list[int]] = []
    while len(prefixes) < n:
        board = chess.Board()
        ids = [
            tok.bos_id,
            tok.vocab["<w1800>"],
            tok.vocab["<b1800>"],
        ]
        plies = rng.randint(min_ply, max_ply)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            ids.append(tok.vocab.get(move.uci(), tok.unk_id))
            board.push(move)
        prefixes.append(ids)
    return prefixes


def validation_prefixes(
    games: Path,
    tok: UciTokenizer,
    n: int = DEFAULT_N,
    block: int = 200,
    seed: int = 0,
) -> list[list[int]]:
    """Token-id prefixes of ``n`` validation positions, cropped exactly as the model sees them."""
    from rukh.eval.legality import sample_positions

    positions = sample_positions(games, n, tok, seed=seed, pool=max(n * 5, 10_000), block=block)
    return [list(position.history) for position in positions]


def parity_positions(
    tok: UciTokenizer,
    n: int = DEFAULT_N,
    block: int = 200,
    seed: int = 0,
    games: Path | None = None,
) -> tuple[list[list[int]], str, str | None]:
    """``(prefixes, source, warning)``: validation positions, or random walks with a warning."""
    from rukh import paths

    parquet = Path(games) if games is not None else paths.resolve(DEFAULT_GAMES)
    if parquet.is_file():
        prefixes = validation_prefixes(parquet, tok, n=n, block=block, seed=seed)
        if prefixes:
            return prefixes, "validation", None
    warning = (
        f"no validation games at {parquet}: parity was measured on random legal walks, which "
        "visit positions no human would reach"
    )
    log.warning("%s", warning)
    return random_prefixes(tok, n, seed=seed), "random-walk", warning


def _session(onnx_path: Path):  # type: ignore[no-untyped-def]
    """An ONNX Runtime session on the CPU provider (tests monkeypatch this)."""
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def parity(
    ckpt: Path | MoveDecoder,
    onnx_path: Path,
    positions: Sequence[Sequence[int]],
    n: int = DEFAULT_N,
) -> ParityResult:
    """Compare the argmax token and the logits of the checkpoint and the exported file."""
    model = ckpt if isinstance(ckpt, MoveDecoder) else _load(Path(ckpt))
    wrapper = LastStepLogits(model.eval()).eval()
    session = _session(Path(onnx_path))
    used = list(positions)[:n]
    if not used:
        raise ValueError("parity needs at least one position")

    agreed = 0
    worst = 0.0
    mismatches: list[int] = []
    for index, history in enumerate(used):
        idx = np.asarray([list(history)], dtype=np.int64)
        with torch.no_grad():
            reference = wrapper(torch.from_numpy(idx)).numpy()[0].astype(np.float64)
        exported = np.asarray(session.run(None, {INPUT_NAME: idx})[0])[0].astype(np.float64)
        worst = max(worst, float(np.max(np.abs(reference - exported))))
        if int(np.argmax(reference)) == int(np.argmax(exported)):
            agreed += 1
        elif len(mismatches) < MAX_MISMATCHES:
            mismatches.append(index)
    return ParityResult(
        positions=len(used),
        agreement=agreed / len(used),
        max_abs_logit_delta=worst,
        mismatches=mismatches,
    )


def _load(ckpt: Path) -> MoveDecoder:
    from rukh.train import load_model

    model, _payload = load_model(ckpt)
    return model
