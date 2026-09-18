"""Stockfish: locate the binary and play a short sanity game at a limited ``UCI_Elo``."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import chess
import chess.engine
from pydantic import BaseModel, ConfigDict

from rukh import paths

MOVE_TIME_SECONDS = 0.05


class EngineNotFound(RuntimeError):
    """No Stockfish binary could be located."""


class EngineError(ValueError):
    """The engine started but does not behave like Stockfish (no usable ``UCI_Elo``)."""


class EngineCheckResult(BaseModel):
    """Outcome of ``engine_check``: what engine answered and how the test game went."""

    model_config = ConfigDict(extra="forbid")

    path: str
    name: str
    uci_elo_min: int
    uci_elo_max: int
    elo: int
    plies_played: int
    result: str
    nodes_hint: int | None


def find_stockfish() -> Path | None:
    """Locate Stockfish: ``RUKH_STOCKFISH``, then ``tools/stockfish/``, then ``PATH``.

    When ``RUKH_STOCKFISH`` is set it wins outright and must point at an existing file;
    otherwise ``EngineNotFound`` is raised instead of silently falling through.
    """
    env = os.environ.get("RUKH_STOCKFISH")
    if env:
        candidate = Path(env)
        if not candidate.is_file():
            raise EngineNotFound(f"RUKH_STOCKFISH={env} is not a file")
        return candidate
    tools = paths.tools_dir() / "stockfish"
    for pattern in ("stockfish*.exe", "stockfish"):
        for candidate in sorted(tools.glob(pattern)):
            if candidate.is_file():
                return candidate
    which = shutil.which("stockfish")
    return Path(which) if which else None


def engine_check(elo: int = 1400, plies: int = 40, seed: int = 0) -> EngineCheckResult:
    """Open Stockfish, limit it to ``elo`` and play ``plies`` half-moves against a random mover.

    Stockfish plays White, the seeded random player plays Black. Raises ``EngineNotFound`` when
    no binary is available, ``EngineError`` when the binary does not expose ``UCI_Elo`` and
    ``ValueError`` when ``elo`` is outside the engine's range.
    """
    path = find_stockfish()
    if path is None:
        raise EngineNotFound(
            "Stockfish not found: run `uv run python scripts/get_stockfish.py` "
            "or set RUKH_STOCKFISH"
        )

    rng = random.Random(seed)
    board = chess.Board()
    nodes_hint: int | None = None
    plies_played = 0

    with chess.engine.SimpleEngine.popen_uci(str(path)) as engine:
        name = str(engine.id.get("name", "unknown"))
        try:
            option = engine.options["UCI_Elo"]
            uci_elo_min, uci_elo_max = int(option.min), int(option.max)
        except (
            KeyError,
            TypeError,
            chess.engine.EngineError,
            chess.engine.EngineTerminatedError,
        ) as exc:
            raise EngineError(f"{path} ({name}) does not expose UCI_Elo; is it Stockfish?") from exc
        if not uci_elo_min <= elo <= uci_elo_max:
            raise ValueError(f"UCI_Elo {elo} outside [{uci_elo_min}, {uci_elo_max}]")
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

        while plies_played < plies and not board.is_game_over(claim_draw=True):
            if board.turn == chess.WHITE:
                played = engine.play(
                    board, chess.engine.Limit(time=MOVE_TIME_SECONDS), info=chess.engine.INFO_BASIC
                )
                if played.move is None:
                    break
                move = played.move
                nodes = played.info.get("nodes")
                if isinstance(nodes, int):
                    nodes_hint = nodes
            else:
                move = rng.choice(list(board.legal_moves))
            board.push(move)
            plies_played += 1

    return EngineCheckResult(
        path=str(path),
        name=name,
        uci_elo_min=uci_elo_min,
        uci_elo_max=uci_elo_max,
        elo=elo,
        plies_played=plies_played,
        result=board.result(claim_draw=True),
        nodes_hint=nodes_hint,
    )
