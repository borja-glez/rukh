"""Legality without the mask: the honest "does it understand chess" metric.

A position is a prefix of a validation game: the moves up to a random ply, encoded exactly as
training encodes them (``<bos> <wXXXX> <bXXXX> moves...``). The model is asked for one move with
``mask_illegal=False``, so the raw token comes out as the network produced it, and the share of
tokens that happen to be legal moves in that position is the legality rate of ``docs/spec/02``.

The sampler of validation prefixes lives here because legality is what defines it; ``accuracy``
reuses the same ``Position`` objects (it only needs the human move on top).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import chess
from pydantic import BaseModel, ConfigDict

from rukh.infer import SampleConfig, legal_token_ids, pick_move
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer, elo_token

GAME_COLUMNS = ["game_id", "uci", "white_elo", "black_elo"]
MIN_PLY = 1


class Position(BaseModel):
    """One validation prefix: the board, the ids the model would have seen and the human reply."""

    model_config = ConfigDict(extra="forbid")

    game_id: int
    ply: int
    """Number of half-moves already played (the model must produce half-move ``ply``)."""
    moves: list[str]
    """The prefix in UCI, so the board can be rebuilt without carrying a FEN."""
    history: list[int]
    """Token ids of the prefix, header included."""
    target: str | None
    """The move the human played next, or None at the end of the game."""
    elo: int
    """Rating of the side to move (what the Elo bands of ``accuracy`` are cut on)."""


def board_of(position: Position) -> chess.Board:
    """Replay the prefix into a board."""
    board = chess.Board()
    for uci in position.moves:
        board.push(chess.Move.from_uci(uci))
    return board


def header(tok: UciTokenizer, white_elo: int, black_elo: int) -> list[int]:
    """``[<bos>, <wXXXX>, <bXXXX>]``, the three tokens every training sequence starts with."""
    return [
        tok.bos_id,
        tok.vocab[elo_token(white_elo, "w")],
        tok.vocab[elo_token(black_elo, "b")],
    ]


def position_at(
    tok: UciTokenizer,
    game_id: int,
    moves: Sequence[str],
    ply: int,
    white_elo: int,
    black_elo: int,
    block: int = 200,
) -> Position:
    """Build the prefix of ``moves`` at ``ply``, cropped on the left to ``block`` ids."""
    prefix = list(moves[:ply])
    ids = header(tok, white_elo, black_elo)
    ids.extend(tok.vocab.get(uci, tok.unk_id) for uci in prefix)
    return Position(
        game_id=game_id,
        ply=ply,
        moves=prefix,
        history=ids[-block:],
        target=moves[ply] if ply < len(moves) else None,
        elo=white_elo if ply % 2 == 0 else black_elo,
    )


def sample_positions(
    games: Path,
    n: int,
    tok: UciTokenizer,
    seed: int = 0,
    pool: int = 50_000,
    block: int = 200,
) -> list[Position]:
    """Sample ``n`` prefixes from a UCI games parquet, one per game, at a random ply.

    Only the first ``pool`` rows are read (a validation month is millions of games and the
    harness never needs more than a few tens of thousands); which of them are used, and at which
    ply, is decided by ``seed``.
    """
    import polars as pl

    frame = pl.scan_parquet(Path(games).as_posix()).select(GAME_COLUMNS).head(pool).collect()
    rows = frame.rows(named=True)
    if not rows:
        return []
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    positions: list[Position] = []
    for index in order:
        if len(positions) >= n:
            break
        row = rows[index]
        moves = str(row["uci"]).split()
        if len(moves) <= MIN_PLY:
            continue
        ply = rng.randrange(MIN_PLY, len(moves))
        positions.append(
            position_at(
                tok,
                int(row["game_id"]),
                moves,
                ply,
                int(row["white_elo"]),
                int(row["black_elo"]),
                block=block,
            )
        )
    return positions


class LegalityResult(BaseModel):
    """Share of unmasked proposals that were legal moves."""

    model_config = ConfigDict(extra="forbid")

    positions: int
    legal: int
    rate: float


def legality(
    model: MoveDecoder,
    tok: UciTokenizer,
    positions: Sequence[Position],
    cfg: SampleConfig | None = None,
) -> LegalityResult:
    """Ask the model for one unmasked move per position and count the legal ones."""
    sample = (cfg or SampleConfig()).model_copy(update={"mask_illegal": False})
    generator = sample.generator()
    legal = 0
    counted = 0
    for position in positions:
        board = board_of(position)
        if not legal_token_ids(board, tok):
            continue
        counted += 1
        _, report = pick_move(model, tok, board, position.history, sample, generator)
        legal += int(report["legal"])
    return LegalityResult(
        positions=counted,
        legal=legal,
        rate=legal / counted if counted else 0.0,
    )
