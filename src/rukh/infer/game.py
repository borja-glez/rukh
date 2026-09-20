"""Playing a whole game: a player against an opponent, counting illegal proposals.

An opponent is anything with ``choose(board) -> chess.Move``: the seeded ``RandomOpponent``
used by the unit tests, ``StockfishOpponent`` for the Elo harness, or the browser in the demo.

The *player* -- the side being measured -- is a protocol too, and for one reason: M4 has to put
a general language model through the same ladder as the decoder, and a comparison where the two
sides are measured by different code is not a comparison. ``DecoderPlayer`` is the original path,
unchanged and still the default; ``rukh.eval.qwen_source`` supplies the other one. Everything
around them -- the opponent, termination, adjudication, how a cut game is scored -- stays here,
which is exactly the part that must not differ between the two.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import TracebackType
from typing import Protocol

import chess
import chess.engine
from pydantic import BaseModel, ConfigDict

from rukh.infer.sampler import (
    HEADER_TOKENS,
    SampleConfig,
    legal_token_ids,
    model_generator,
    pick_move,
)
from rukh.models import MoveDecoder
from rukh.tokenize.uci_vocab import RESULT_TOKENS, UciTokenizer, elo_token

MOVE_TIME_SECONDS = 0.1
"""Default seconds per move for the Stockfish opponent (``docs/decisiones-de-ejecucion`` D-025)."""
REPETITION_CLOCK = 8
"""A threefold repetition needs at least this many reversible plies, so the check is gated on it."""
FIFTY_MOVE_PLIES = 100


class Opponent(Protocol):
    """Whatever plays the other side."""

    def choose(self, board: chess.Board) -> chess.Move: ...


class Player(Protocol):
    """Whatever plays the side being measured.

    ``choose`` returns the move the player will actually make **and** whether its unmasked
    proposal was legal. The two are different questions: a game has to keep going, so an illegal
    proposal is rescued with a masked draw, but the rescue must not hide the fact that it
    happened -- that count is the legality signal the whole harness is built around.
    """

    @property
    def limit(self) -> int:
        """Plies this player can carry before it runs out of context."""

    def start(self, white_elo: int, black_elo: int) -> None:
        """Begin a new game at the given ratings."""

    def choose(self, board: chess.Board) -> tuple[chess.Move | None, bool]: ...

    def observe(self, move: chess.Move) -> None:
        """Record a move that was played, by either side."""


class DecoderPlayer:
    """``MoveDecoder`` as a ``Player``: the path every published number was measured on."""

    def __init__(self, model: MoveDecoder, tok: UciTokenizer, cfg: SampleConfig) -> None:
        self.model, self.tok, self.cfg = model, tok, cfg
        self.masked = cfg.model_copy(update={"mask_illegal": True})
        self.generator = model_generator(model, cfg)
        self.history: list[int] = []

    @property
    def limit(self) -> int:
        return self.model.cfg.block - HEADER_TOKENS - 1

    def start(self, white_elo: int, black_elo: int) -> None:
        self.history = _history(self.tok, white_elo, black_elo)

    def choose(self, board: chess.Board) -> tuple[chess.Move | None, bool]:
        move, _ = pick_move(self.model, self.tok, board, self.history, self.cfg, self.generator)
        if move is not None:
            return move, True
        rescued, _ = pick_move(
            self.model, self.tok, board, self.history, self.masked, self.generator
        )
        return rescued, False

    def observe(self, move: chess.Move) -> None:
        self.history.append(self.tok.vocab.get(move.uci(), self.tok.unk_id))


class RandomOpponent:
    """A seeded random legal mover: the cheap baseline every test can afford."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose(self, board: chess.Board) -> chess.Move:
        return self.rng.choice(list(board.legal_moves))


class StockfishOpponent:
    """Stockfish limited to ``elo``; close it (or use it as a context manager) when done.

    ``UCI_Elo`` bottoms out at 1320, so a ``skill`` between 0 and 20 may be given instead: it
    selects ``Skill Level``, the only way to get rungs below that floor for the Elo harness.
    """

    def __init__(
        self,
        elo: int = 1400,
        move_time: float = MOVE_TIME_SECONDS,
        path: Path | None = None,
        skill: int | None = None,
    ) -> None:
        from rukh.engine import EngineNotFound, find_stockfish

        binary = path or find_stockfish()
        if binary is None:
            raise EngineNotFound("Stockfish not found; set RUKH_STOCKFISH or run get_stockfish.py")
        self.move_time = move_time
        self.engine = chess.engine.SimpleEngine.popen_uci(str(binary))
        if skill is None:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        else:
            self.engine.configure({"UCI_LimitStrength": False, "Skill Level": skill})

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
    fen: str = chess.STARTING_FEN
    """The final position, so a game cut by the context limit can still be adjudicated."""

    @property
    def cut(self) -> bool:
        """True when the game ran out of context instead of ending on the board."""
        return self.result == "*"


def is_over(board: chess.Board) -> bool:
    """Whether the game is finished, including the claimable draws, without the O(n^2) check.

    ``board.is_game_over(claim_draw=True)`` tries every legal move looking for a claimable
    repetition, which is quadratic over a long game and is called once per ply. The claimable
    draws are covered here by the fifty-move clock and by ``is_repetition`` (a scan of the move
    stack) gated on a halfmove clock that makes a repetition possible at all.
    """
    if board.is_game_over():
        return True
    if board.halfmove_clock >= FIFTY_MOVE_PLIES:
        return True
    return board.halfmove_clock >= REPETITION_CLOCK and board.is_repetition(3)


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

    A game that hits ``max_plies`` comes back with ``result="*"`` and the final ``fen``: scoring
    it as a draw would flatter (or punish) the model, so the Elo harness adjudicates it instead.
    """
    player = DecoderPlayer(model, tok, cfg)
    return play_game_with(
        player,
        opponent,
        model_color=model_color,
        white_elo=white_elo,
        black_elo=black_elo,
        max_plies=max_plies,
        board=board,
    )


def _tell(opponent: Opponent, move: chess.Move) -> None:
    """Show a move to an opponent that keeps state; a no-op for one that does not."""
    observe = getattr(opponent, "observe", None)
    if callable(observe):
        observe(move)


def play_game_with(
    player: Player,
    opponent: Opponent,
    model_color: chess.Color = chess.WHITE,
    white_elo: int = 1800,
    black_elo: int = 1800,
    max_plies: int | None = None,
    board: chess.Board | None = None,
) -> GameResult:
    """``play_game`` for any ``Player``; the decoder's version is the one-line wrapper above."""
    board = board if board is not None else chess.Board()
    limit = max_plies if max_plies is not None else player.limit
    player.start(white_elo, black_elo)
    # A board handed in with moves already on it is a position the player has to *know about*:
    # its prompt is the game so far, not the game from here. Without this a match that starts
    # from an opening book plays every move blind to the opening, which reads as a model that
    # forgot how to play. The match harness of P5 found it with a model against itself.
    for played in board.move_stack:
        player.observe(played)
        _tell(opponent, played)
    moves: list[str] = []
    illegal = 0

    while len(moves) < limit and not is_over(board):
        if board.turn == model_color:
            move, was_legal = player.choose(board)
            if not was_legal:
                illegal += 1
            if move is None:
                break
        else:
            move = opponent.choose(board)
        moves.append(move.uci())
        player.observe(move)
        # An opponent that keeps state -- another model, in a head-to-head match -- has to see
        # the moves too. Stockfish and the random mover have no `observe` and are unaffected.
        _tell(opponent, move)
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    return GameResult(
        result=outcome.result() if outcome else "*",
        plies=len(moves),
        illegal_proposals=illegal,
        moves=moves,
        termination=outcome.termination.name.lower() if outcome else "cut_short",
        fen=board.fen(),
    )


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}
ADJUDICATION_DEPTH = 8
ADJUDICATION_CP = 200
"""Centipawn margin above which an adjudicated position counts as a win."""
ADJUDICATION_PAWNS = ADJUDICATION_CP / 100.0
MATE_SCORE = 10_000


def material_balance(board: chess.Board) -> float:
    """Material of the position in pawns, from White's point of view."""
    total = 0.0
    for piece_type, value in PIECE_VALUES.items():
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total -= value * len(board.pieces(piece_type, chess.BLACK))
    return total


def adjudicate(
    board: chess.Board | str,
    engine: chess.engine.SimpleEngine | None = None,
    depth: int = ADJUDICATION_DEPTH,
    margin_cp: int = ADJUDICATION_CP,
) -> tuple[str, str]:
    """Decide a game that was cut short; returns ``(result, how)``.

    With an engine the final position is analysed to a shallow depth and a score of at least
    ``margin_cp`` for one side is a win for that side; without one (or when the analysis gives no
    centipawn score) the same margin is applied to the material count, which is crude but is
    still an answer, and is far better than booking every cut game as half a point.
    """
    position = chess.Board(board) if isinstance(board, str) else board
    if engine is not None:
        try:
            info = engine.analyse(position, chess.engine.Limit(depth=depth))
            score = info["score"].white().score(mate_score=MATE_SCORE)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, KeyError):
            score = None
        if score is not None:
            return _verdict(float(score), float(margin_cp)), f"engine depth {depth}"
    pawns = material_balance(position)
    return _verdict(pawns * 100.0, float(margin_cp)), "material count"


def _verdict(centipawns: float, margin: float) -> str:
    """``1-0``/``0-1``/``1/2-1/2`` from a White-relative score in centipawns."""
    if centipawns >= margin:
        return "1-0"
    if centipawns <= -margin:
        return "0-1"
    return "1/2-1/2"


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
    """Share of the positions actually queried where the unmasked network proposes a legal move.

    Positions with no legal move (checkmate, stalemate) are never asked, so they are not counted
    in the denominator either: the rate is over the boards the model was really given.
    """
    if not boards:
        return 0.0
    unmasked = cfg.model_copy(update={"mask_illegal": False})
    generator = model_generator(model, cfg)
    legal = 0
    counted = 0
    for board, history in zip(boards, histories, strict=True):
        if not legal_token_ids(board, tok):
            continue
        counted += 1
        _, report = pick_move(model, tok, board, history, unmasked, generator)
        legal += int(report["legal"])
    return legal / counted if counted else 0.0
