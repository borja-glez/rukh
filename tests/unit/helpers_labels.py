"""Rows of ``positions-eval.parquet`` for the tests that build labels out of a real game.

Not a ``conftest.py`` and not a package: ``tests/`` has no ``__init__.py`` on purpose (pytest
inserts ``tests/unit`` on ``sys.path``, so ``from helpers_labels import ...`` works and
``from tests.unit...`` does not), and these are plain constructors rather than fixtures because
they are called with arguments and outside of a test as often as inside one.
"""

from __future__ import annotations

from typing import Any

import chess
import polars as pl

GAME = ("e4", "e5", "Qh5", "Nc6", "Qxf7+", "Kxf7", "Nf3", "Nf6")
"""1.e4 e5 2.Qh5 Nc6 3.Qxf7+?? Kxf7: a real queen blunder inside a real game."""


def game_rows(game_id: int, sans: tuple[str, ...], scores: list[int]) -> list[dict[str, Any]]:
    """One row of ``positions-eval.parquet`` per ply of a game actually played on a board."""
    board = chess.Board()
    rows: list[dict[str, Any]] = []
    for ply, san in enumerate(sans, start=1):
        move = board.parse_san(san)
        board.push(move)
        rows.append(
            {
                "fen": " ".join(board.fen().split(" ")[:4]),
                "game_id": game_id,
                "ply": ply,
                "last_move": move.uci(),
                "result": "0-1",
                "phase": "opening",
                "cp": scores[ply - 1],
                "mate": None,
                "n_seen": 1,
                "best_move": move.uci(),
                "depth": 20,
            }
        )
    return rows


def source_frame() -> pl.DataFrame:
    """Two copies of the same game, so both sides of the by-game split have rows."""
    scores = [20, 10, 30, 15, -850, -900, -880, -870]
    return pl.DataFrame(game_rows(1, GAME, scores) + game_rows(2, GAME, scores))


def game_moves_frame() -> pl.DataFrame:
    """The ``data/uci`` rows matching ``source_frame``: one ``uci`` string per game."""
    board = chess.Board()
    uci = []
    for san in GAME:
        move = board.parse_san(san)
        board.push(move)
        uci.append(move.uci())
    line = " ".join(uci)
    return pl.DataFrame(
        {
            "game_id": [1, 2],
            "uci": [line, line],
            "n_plies": [len(uci), len(uci)],
            "white_elo": [1850, 1850],
            "black_elo": [1900, 1900],
            "result": ["0-1", "0-1"],
        }
    )
