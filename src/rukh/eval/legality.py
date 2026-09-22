"""Legality without the mask: the honest "does it understand chess" metric.

A position is a prefix of a validation game: the moves up to a random ply, encoded exactly as
training encodes them (``<bos> <wXXXX> <bXXXX> moves...``). The model is asked for one move with
``mask_illegal=False``, so the raw token comes out as the network produced it, and the share of
tokens that happen to be legal moves in that position is the legality rate of the design spec 02.

It is measured twice, because the two numbers answer different questions:

``argmax``
    the single most likely token, with no temperature and no top-k. This is the headline rate
    and the one the "at least 99 % legal" bar of ``docs/acceptance.md`` refers to: it is a property
    of the
    weights, not of a sampler setting.
``sampled``
    a token drawn exactly as the demo draws it (the suite's temperature and top-k). It is
    always the lower of the two and is what a player would actually experience without a mask.

The sampler of validation prefixes lives here because legality is what defines it; ``accuracy``
reuses the same ``Position`` objects (it only needs the human move on top).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import chess
from pydantic import BaseModel, ConfigDict

from rukh.infer import (
    SampleConfig,
    legal_token_ids,
    model_generator,
    pick_move,
    prompt_ids,
)
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer, elo_token

LEGALITY_MODES = ("argmax", "sampled")

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
    """Build the prefix of ``moves`` at ``ply``, cropped to ``block`` ids header-first.

    The crop keeps ``<bos> <wXXXX> <bXXXX>`` and drops the oldest moves (``prompt_ids``): a plain
    left crop would take the Elo conditioning away from every long prefix.
    """
    prefix = list(moves[:ply])
    ids = header(tok, white_elo, black_elo)
    ids.extend(tok.vocab.get(uci, tok.unk_id) for uci in prefix)
    return Position(
        game_id=game_id,
        ply=ply,
        moves=prefix,
        history=prompt_ids(ids, block),
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
    header_elo: int | None = None,
) -> list[Position]:
    """Sample ``n`` prefixes from a UCI games parquet, one per game, at a random ply.

    Only the first ``pool`` rows are read (a validation month is millions of games and the
    harness never needs more than a few tens of thousands); which of them are used, and at which
    ply, is decided by ``seed``.

    ``header_elo`` replaces the ratings the game really had. It is off by default, because the
    honest way to score next-move accuracy is to show the model the header the game actually
    carried. It is on for the Elo sweep, and for the same reason the puzzles need it: every
    position is prompted with the *condition* being tested, so "asked to play at 1200, does it
    play worse or does it play illegally?" becomes a question the table can answer. Without it
    those two columns are identical in every row -- true, and indistinguishable from evidence
    that the condition does nothing.
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
                header_elo if header_elo is not None else int(row["white_elo"]),
                header_elo if header_elo is not None else int(row["black_elo"]),
                block=block,
            )
        )
    return positions


class LegalityResult(BaseModel):
    """Share of unmasked proposals that were legal moves, and how they were drawn."""

    model_config = ConfigDict(extra="forbid")

    positions: int
    legal: int
    rate: float
    mode: str = "sampled"
    """``argmax`` (the headline definition) or ``sampled`` (the demo's temperature and top-k)."""
    temperature: float | None = None
    top_k: int | None = None


def legality(
    model: MoveDecoder,
    tok: UciTokenizer,
    positions: Sequence[Position],
    cfg: SampleConfig | None = None,
    mode: str = "sampled",
) -> LegalityResult:
    """Ask the model for one unmasked move per position and count the legal ones.

    ``mode="argmax"`` ignores the temperature and the top-k of ``cfg`` and takes the single most
    likely token, which is the definition behind the 99 % bar; ``mode="sampled"`` draws the way
    the demo does.
    """
    if mode not in LEGALITY_MODES:
        raise ValueError(f"unknown legality mode {mode!r}; expected one of {LEGALITY_MODES}")
    update: dict[str, object] = {"mask_illegal": False}
    if mode == "argmax":
        update |= {"temperature": 0.0, "top_k": None}
    sample = (cfg or SampleConfig()).model_copy(update=update)
    generator = model_generator(model, sample)
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
        mode=mode,
        temperature=None if mode == "argmax" else sample.temperature,
        top_k=None if mode == "argmax" else sample.top_k,
    )
