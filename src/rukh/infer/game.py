"""Playing a whole game: the model against an opponent, counting illegal proposals.

An opponent is anything with ``choose(board) -> chess.Move``: the seeded ``RandomOpponent``
used by the unit tests, ``StockfishOpponent`` for the Elo harness, or the browser in the demo.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import TracebackType
from typing import Protocol

import chess
import chess.engine
from pydantic import BaseModel, ConfigDict

from rukh.infer.sampler import SampleConfig, legal_token_ids, pick_move
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import RESULT_TOKENS, UciTokenizer, elo_token

MOVE_TIME_SECONDS = 0.05
HEADER_TOKENS = 3  # <bos> <wXXXX> <bXXXX>


class Opponent(Protocol):
    """Whatever plays the other side."""

    def choose(self, board: chess.Board) -> chess.Move: ...


class RandomOpponent:
    """A seeded random legal mover: the cheap baseline every test can afford."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose(self, board: chess.Board) -> chess.Move:
        return self.rng.choice(list(board.legal_moves))


class StockfishOpponent:
    """Stockfish limited to ``elo``; close it (or use it as a context manager) when done."""

    def __init__(
        self,
        elo: int = 1400,
        move_time: float = MOVE_TIME_SECONDS,
        path: Path | None = None,
    ) -> None:
        from rukh.engine import EngineNotFound, find_stockfish

        binary = path or find_stockfish()
        if binary is None:
            raise EngineNotFound("Stockfish not found; set RUKH_STOCKFISH or run get_stockfish.py")
        self.move_time = move_time
        self.engine = chess.engine.SimpleEngine.popen_uci(str(binary))
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def choose(self, board: chess.Board) -> chess.Move:
        played = self.engine.play(board, chess.engine.Limit(time=self.move_time))
        if played.move is None:
            raise chess.engine.EngineError("Stockfish returned no move")
        return played.move

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self) -> StockfishOpponent:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class GameResult(BaseModel):
    """One finished (or cut) game, from the point of view of the board."""

    model_config = ConfigDict(extra="forbid")

    result: str  # 1-0, 0-1, 1/2-1/2 or * when the game was cut short
    plies: int
    illegal_proposals: int
    moves: list[str]
    termination: str


def _history(tok: UciTokenizer, white_elo: int, black_elo: int) -> list[int]:
    return [
        tok.bos_id,
        tok.vocab[elo_token(white_elo, "w")],
        tok.vocab[elo_token(black_elo, "b")],
    ]


def play_game(
    model: MoveDecoder,
    tok: UciTokenizer,
    opponent: Opponent,
    cfg: SampleConfig,
    model_color: chess.Color = chess.WHITE,
    white_elo: int = 1800,
    black_elo: int = 1800,
    max_plies: int | None = None,
    board: chess.Board | None = None,
) -> GameResult:
    """Play until the game ends or ``max_plies`` is reached and report what happened.

    The model's illegal proposals are counted and then rescued with a masked draw, so a game
    always terminates: ``illegal_proposals`` is the raw legality signal, not a failure.
    """
    board = board if board is not None else chess.Board()
    limit = max_plies if max_plies is not None else model.cfg.block - HEADER_TOKENS - 1
    generator = cfg.generator()
    masked = cfg.model_copy(update={"mask_illegal": True})
    history = _history(tok, white_elo, black_elo)
    moves: list[str] = []
    illegal = 0

    while len(moves) < limit and not board.is_game_over(claim_draw=True):
        if board.turn == model_color:
            move, report = pick_move(model, tok, board, history, cfg, generator)
            if move is None:
                illegal += 1
                move, _ = pick_move(model, tok, board, history, masked, generator)
            if move is None:
                break
        else:
            move = opponent.choose(board)
        uci = move.uci()
        moves.append(uci)
        history.append(tok.vocab.get(uci, tok.unk_id))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    return GameResult(
        result=outcome.result() if outcome else "*",
        plies=len(moves),
        illegal_proposals=illegal,
        moves=moves,
        termination=outcome.termination.name.lower() if outcome else "cut_short",
    )


def result_token(tok: UciTokenizer, result: str) -> int | None:
    """Vocabulary id of a finished game's result token, or None for an unfinished game."""
    name = RESULT_TOKENS.get(result)
    return tok.vocab[name] if name else None


def legality_rate(
    model: MoveDecoder,
    tok: UciTokenizer,
    boards: list[chess.Board],
    histories: list[list[int]],
    cfg: SampleConfig,
) -> float:
    """Share of positions where the unmasked network proposes a legal move."""
    if not boards:
        return 0.0
    unmasked = cfg.model_copy(update={"mask_illegal": False})
    generator = cfg.generator()
    legal = 0
    for board, history in zip(boards, histories, strict=True):
        if not legal_token_ids(board, tok):
            continue
        _, report = pick_move(model, tok, board, history, unmasked, generator)
        legal += int(report["legal"])
    return legal / len(boards)
