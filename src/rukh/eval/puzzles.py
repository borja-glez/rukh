"""Tactical puzzles: a puzzle counts only when the whole solution line is played.

Lichess puzzles are ``fen`` (the position *before* the opponent's last move) plus ``moves``:
the opponent's move first, then the solution, alternating. The opponent's replies are forced, so
they are simply pushed; every move at an odd index is the model's turn and must match exactly.
A wrong move ends the attempt, and ``correct`` records how far down the line it got, which is
what makes a partially solved puzzle distinguishable from a first-move miss.

A move-sequence model cannot be handed a FEN, so what it is prompted with is the only question
that matters here. With ``prefix_uci`` in the parquet (``rukh data puzzles`` with ``with_games``)
the prompt is the real game up to the puzzle position, headed by ``<bos>`` and the two Elo tokens
of the players, exactly as training encodes a game: ``game-prefix``. Without it the only thing
left is the puzzle line itself starting from ``<bos>``, a token sequence that is not a game and
does not start from the initial position: ``line-only``, kept so an old parquet still evaluates,
and recorded in the result because the two numbers are not comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chess
from pydantic import BaseModel, ConfigDict, Field

from rukh.eval.cache import EvalCache
from rukh.eval.legality import header
from rukh.infer import SampleConfig, model_generator, pick_move, prompt_ids
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

PUZZLE_COLUMNS = ["puzzle_id", "fen", "moves", "rating", "band", "split"]
PREFIX_COLUMNS = ["prefix_uci", "white_elo", "black_elo"]
"""Columns of the with-games parquet; absent from a parquet built by the puzzle-only source."""
HEADER_ELO = 1800
"""Elo written in the header when the players' own ratings are not in the parquet."""
GAME_PREFIX = "game-prefix"
LINE_ONLY = "line-only"
MIXED = "mixed"
SUITE = "puzzles"


class PuzzleItem(BaseModel):
    """One puzzle as the harness needs it."""

    model_config = ConfigDict(extra="forbid")

    puzzle_id: str
    fen: str
    moves: list[str]
    rating: int
    band: str
    prefix: list[str] = Field(default_factory=list)
    """The real game's moves up to ``fen``, empty when the parquet does not carry them."""
    white_elo: int | None = None
    black_elo: int | None = None

    @property
    def prompt_style(self) -> str:
        """``game-prefix`` when there is a real game to prompt with, else ``line-only``."""
        return GAME_PREFIX if self.prefix else LINE_ONLY


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
    prompt_style: str = LINE_ONLY
    """How the model was prompted; the default is what every attempt cached before P3 used."""


def start_history(
    tok: UciTokenizer,
    item: PuzzleItem,
    header_elo: int = HEADER_ELO,
    force_header: bool = False,
) -> list[int]:
    """The ids the model sees before its first move: header, then the real game if there is one.

    The header is the ``<bos> <wXXXX> <bXXXX>`` of ``rukh.infer.sampler.prompt_ids``, with the
    players' own ratings when the parquet carries them; ``header_elo`` stands in otherwise.

    ``force_header`` overrides the real ratings, and exists for exactly one measurement: the Elo
    sweep of M4, which asks the same model to play at several strengths and needs *every* column
    to answer that question. Almost all puzzles carry their players' ratings, so without this the
    puzzle rate would be identical in every row of the sweep and would look like evidence that
    the condition does nothing.
    """
    if force_header:
        white = black = header_elo
    else:
        white = item.white_elo if item.white_elo is not None else header_elo
        black = item.black_elo if item.black_elo is not None else header_elo
    history = header(tok, white, black)
    history.extend(tok.vocab.get(uci, tok.unk_id) for uci in item.prefix)
    return history


def solve_puzzle(
    source: MoveSource,
    tok: UciTokenizer,
    item: PuzzleItem,
    header_elo: int = HEADER_ELO,
    block: int = 200,
    force_header: bool = False,
) -> PuzzleAttempt:
    """Play the puzzle line; stop at the first move that is not the expected one.

    The board comes from the puzzle's FEN, which is authoritative; the prefix only builds the
    token history, because that is all a move-sequence model reads.
    """
    board = chess.Board(item.fen)
    history = start_history(tok, item, header_elo, force_header)
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
        prompt_style=item.prompt_style,
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
    prompt_style: str = LINE_ONLY
    """``game-prefix``, ``line-only`` or ``mixed``: what the model was actually asked."""

    def by_band(self) -> dict[str, float]:
        return {band.band: band.rate for band in self.bands}


def run_puzzles(
    source: MoveSource,
    tok: UciTokenizer,
    items: Sequence[PuzzleItem],
    cache: EvalCache | None = None,
    header_elo: int = HEADER_ELO,
    force_header: bool = False,
) -> PuzzleResult:
    """Attempt every puzzle, reusing the cached attempts of a previous run of the same weights."""
    attempts: list[PuzzleAttempt] = []
    for item in items:
        # A puzzle whose parquet carries the players' own ratings ignores ``header_elo``, so its
        # key stays as it was; one that falls back to the header is a different attempt per
        # header and must not read back the 1800 run's answer.
        uses_header = force_header or item.white_elo is None or item.black_elo is None
        key = item.puzzle_id
        if uses_header and header_elo != HEADER_ELO:
            key = f"{key}:e{header_elo}"
        cached = cache.get(SUITE, key) if cache is not None else None
        if cached is not None:
            attempt = PuzzleAttempt.model_validate(cached)
            # An attempt played with the other prompt answers a different question.
            if attempt.prompt_style == item.prompt_style:
                attempts.append(attempt)
                continue
        attempt = solve_puzzle(source, tok, item, header_elo=header_elo, force_header=force_header)
        if cache is not None:
            cache.put(SUITE, key, attempt.model_dump())
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
    styles = {attempt.prompt_style for attempt in attempts}
    return PuzzleResult(
        prompt_style=styles.pop() if len(styles) == 1 else (MIXED if styles else LINE_ONLY),
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
    """Read up to ``per_band`` puzzles of the given split, with the game prefix when it is there."""
    import polars as pl

    scan = pl.scan_parquet(Path(path).as_posix())
    present = set(scan.collect_schema().names())
    columns = PUZZLE_COLUMNS + [name for name in PREFIX_COLUMNS if name in present]
    frame = scan.select(columns).filter(pl.col("split") == split).collect()
    items: list[PuzzleItem] = []
    for band in sorted(frame["band"].unique().to_list()):
        rows = (
            frame.filter(pl.col("band") == band)
            .sample(n=min(per_band, frame.filter(pl.col("band") == band).height), seed=seed)
            .rows(named=True)
        )
        items.extend(_item(row) for row in rows)
    return items


def _item(row: dict[str, object]) -> PuzzleItem:
    """One parquet row as a ``PuzzleItem``; the prefix columns are optional."""
    return PuzzleItem(
        puzzle_id=str(row["puzzle_id"]),
        fen=str(row["fen"]),
        moves=str(row["moves"]).split(),
        rating=int(row["rating"]),  # type: ignore[arg-type]
        band=str(row["band"]),
        prefix=str(row.get("prefix_uci") or "").split(),
        white_elo=_elo(row.get("white_elo")),
        black_elo=_elo(row.get("black_elo")),
    )


def _elo(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None
