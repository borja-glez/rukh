"""Tactical puzzles: a puzzle counts only when the whole solution line is played.

Lichess puzzles are ``fen`` (the position *before* the opponent's last move) plus ``moves``:
the opponent's move first, then the solution, alternating. The opponent's replies are forced, so
they are simply pushed; every move at an odd index is the model's turn and must match exactly.
A wrong move ends the attempt, and ``correct`` records how far down the line it got, which is
what makes a partially solved puzzle distinguishable from a first-move miss.

The model only ever sees the moves of the puzzle line, never the game that produced the
position: a move-sequence model cannot be handed a FEN. The header is a fixed Elo pair, so the
prompt looks like the start of a game between two players of that strength. It is a handicap the
number has to be read with, not a bug.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chess
from pydantic import BaseModel, ConfigDict

from rukh.eval.cache import EvalCache
from rukh.eval.legality import header
from rukh.infer import SampleConfig, model_generator, pick_move, prompt_ids
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

PUZZLE_COLUMNS = ["puzzle_id", "fen", "moves", "rating", "band", "split"]
HEADER_ELO = 1800
SUITE = "puzzles"


class PuzzleItem(BaseModel):
    """One puzzle as the harness needs it."""

    model_config = ConfigDict(extra="forbid")

    puzzle_id: str
    fen: str
    moves: list[str]
    rating: int
    band: str


class MoveSource(Protocol):
    """Anything that answers one move for a position: the model, or a script in the tests."""

    def __call__(self, board: chess.Board, history: list[int]) -> chess.Move | None: ...


def model_source(
    model: MoveDecoder, tok: UciTokenizer, cfg: SampleConfig | None = None
) -> MoveSource:
    """A ``MoveSource`` backed by the masked sampler (temperature 0 by default: argmax)."""
    sample = (cfg or SampleConfig(temperature=0.0)).model_copy(update={"mask_illegal": True})
    generator = model_generator(model, sample)

    def choose(board: chess.Board, history: list[int]) -> chess.Move | None:
        move, _ = pick_move(model, tok, board, history, sample, generator)
        return move

    return choose


class PuzzleAttempt(BaseModel):
    """What happened on one puzzle."""

    model_config = ConfigDict(extra="forbid")

    puzzle_id: str
    band: str
    rating: int
    solved: bool
    correct: int
    """Model moves played correctly before the first mistake (or the whole line)."""
    total: int
    """Model moves the line asks for."""


def solve_puzzle(
    source: MoveSource,
    tok: UciTokenizer,
    item: PuzzleItem,
    header_elo: int = HEADER_ELO,
    block: int = 200,
) -> PuzzleAttempt:
    """Play the puzzle line; stop at the first move that is not the expected one."""
    board = chess.Board(item.fen)
    history = header(tok, header_elo, header_elo)
    expected = [uci for index, uci in enumerate(item.moves) if index % 2 == 1]
    correct = 0
    for index, uci in enumerate(item.moves):
        move = chess.Move.from_uci(uci)
        if index % 2 == 1:
            played = source(board, prompt_ids(history, block))
            if played is None or played.uci() != uci:
                break
            correct += 1
        board.push(move)
        history.append(tok.vocab.get(uci, tok.unk_id))
    return PuzzleAttempt(
        puzzle_id=item.puzzle_id,
        band=item.band,
        rating=item.rating,
        solved=correct == len(expected) and bool(expected),
        correct=correct,
        total=len(expected),
    )


class BandPuzzles(BaseModel):
    """Solved rate inside one difficulty band."""

    model_config = ConfigDict(extra="forbid")

    band: str
    attempted: int
    solved: int
    rate: float


class PuzzleResult(BaseModel):
    """Solved rate overall and per band."""

    model_config = ConfigDict(extra="forbid")

    attempted: int
    solved: int
    rate: float
    bands: list[BandPuzzles]

    def by_band(self) -> dict[str, float]:
        return {band.band: band.rate for band in self.bands}


def run_puzzles(
    source: MoveSource,
    tok: UciTokenizer,
    items: Sequence[PuzzleItem],
    cache: EvalCache | None = None,
    header_elo: int = HEADER_ELO,
) -> PuzzleResult:
    """Attempt every puzzle, reusing the cached attempts of a previous run of the same weights."""
    attempts: list[PuzzleAttempt] = []
    for item in items:
        cached = cache.get(SUITE, item.puzzle_id) if cache is not None else None
        if cached is not None:
            attempts.append(PuzzleAttempt.model_validate(cached))
            continue
        attempt = solve_puzzle(source, tok, item, header_elo=header_elo)
        if cache is not None:
            cache.put(SUITE, item.puzzle_id, attempt.model_dump())
        attempts.append(attempt)
    return summarize(attempts)


def summarize(attempts: Sequence[PuzzleAttempt]) -> PuzzleResult:
    """Fold attempts into overall and per-band rates."""
    counts: dict[str, int] = {}
    solved: dict[str, int] = {}
    for attempt in attempts:
        counts[attempt.band] = counts.get(attempt.band, 0) + 1
        solved[attempt.band] = solved.get(attempt.band, 0) + int(attempt.solved)
    total = sum(counts.values())
    return PuzzleResult(
        attempted=total,
        solved=sum(solved.values()),
        rate=sum(solved.values()) / total if total else 0.0,
        bands=[
            BandPuzzles(
                band=band,
                attempted=counts[band],
                solved=solved.get(band, 0),
                rate=solved.get(band, 0) / counts[band],
            )
            for band in sorted(counts)
        ],
    )


def load_puzzles(path: Path, per_band: int, seed: int = 0, split: str = "test") -> list[PuzzleItem]:
    """Read up to ``per_band`` puzzles of the given split from the P1 puzzle parquet."""
    import polars as pl

    frame = (
        pl.scan_parquet(Path(path).as_posix())
        .select(PUZZLE_COLUMNS)
        .filter(pl.col("split") == split)
        .collect()
    )
    items: list[PuzzleItem] = []
    for band in sorted(frame["band"].unique().to_list()):
        rows = (
            frame.filter(pl.col("band") == band)
            .sample(n=min(per_band, frame.filter(pl.col("band") == band).height), seed=seed)
            .rows(named=True)
        )
        items.extend(
            PuzzleItem(
                puzzle_id=str(row["puzzle_id"]),
                fen=str(row["fen"]),
                moves=str(row["moves"]).split(),
                rating=int(row["rating"]),
                band=str(row["band"]),
            )
            for row in rows
        )
    return items
