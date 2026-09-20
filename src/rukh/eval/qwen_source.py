"""A fine-tuned general model plugged into the harness of P2, so the comparison is like for like.

The decoder cannot write a move that does not exist: its vocabulary *is* the set of moves, and an
"illegal" move is a legal move played in the wrong position. A language model writing SAN can fail
in three ways instead of one, and telling them apart is most of what this comparison teaches:

- **unparseable** -- what it wrote is not a move at all (``Nf9``, ``O-O-O-O``, a sentence);
- **illegal** -- a well-formed move that this position does not allow;
- **ambiguous** -- a move that two pieces could make, which SAN is supposed to disambiguate and
  the model did not.

All three are counted separately and none is silently repaired. A game still has to finish, so a
failed proposal falls back to a legal move, exactly as the decoder's does, and the fallback is
recorded rather than hidden -- same rule as ``illegal_proposals`` in ``rukh.infer.game``.

The prompt is the PGN text of ``rukh.data.pgn_text``, which is the format the model was fine-tuned
on, with the same rating header the decoder gets as ``<wXXXX> <bXXXX>``. Generation is greedy and
six tokens long: a move is one to four of Qwen's tokens and nothing longer is going to help.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import chess
from pydantic import BaseModel, ConfigDict

from rukh.data.pgn_text import prompt_for

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

__all__ = ["QwenPlayer", "QwenStats", "parse_san_answer", "qwen_source"]

MAX_NEW_TOKENS = 6
DECODER_PLIES = 196
"""Plies the decoder can carry (``block`` 200 minus its three header tokens and the move it is
about to make). The text model gets the same number, not the 120 its fine-tuning corpus was cut
at: a harness that let one side play longer games than the other would be measuring the harness.
Past 120 plies the prompt is longer than anything Qwen was fine-tuned on, and that is a property
of the model rather than of the measurement -- it belongs in the result, not in the setup."""
MOVE_NUMBER = re.compile(r"^\d+\.+")
"""``12.`` or ``12...`` at the head of the answer: bookkeeping, not a move."""
TRAILING = "!?"
"""Annotation glyphs a commentator adds; ``+`` and ``#`` are part of SAN and stay."""


class QwenStats(BaseModel):
    """How the answers came out, by failure mode rather than as one number."""

    model_config = ConfigDict(extra="forbid")

    asked: int = 0
    legal: int = 0
    unparseable: int = 0
    """The answer was not a move in any position."""
    illegal: int = 0
    """A well-formed move that this position does not allow."""
    ambiguous: int = 0
    """SAN that two pieces could satisfy; the model failed to disambiguate."""
    empty: int = 0
    """The model produced no non-whitespace token at all."""

    @property
    def legal_rate(self) -> float:
        return self.legal / self.asked if self.asked else 0.0

    def record(self, outcome: str) -> None:
        self.asked += 1
        setattr(self, outcome, getattr(self, outcome) + 1)


def parse_san_answer(board: chess.Board, answer: str) -> tuple[chess.Move | None, str]:
    """``(move, outcome)`` for the model's continuation, with the outcome named.

    Only the first token is considered. A model that writes ``e4 e5 2. Nf3`` has answered the
    question asked and then kept going; scoring the whole continuation would be scoring a
    different question, and taking the last move would be scoring luck.
    """
    text = answer.strip()
    if not text:
        return None, "empty"
    first = text.split()[0]
    stripped = MOVE_NUMBER.sub("", first)
    if not stripped:  # the answer was just a move number; the move is the next token
        parts = text.split()
        if len(parts) < 2:
            return None, "empty"
        stripped = parts[1]
    stripped = stripped.rstrip(TRAILING)
    if not stripped:
        return None, "unparseable"
    try:
        return board.parse_san(stripped), "legal"
    except chess.IllegalMoveError:
        return None, "illegal"
    except chess.AmbiguousMoveError:
        return None, "ambiguous"
    except (chess.InvalidMoveError, ValueError):
        return None, "unparseable"


class QwenPlayer:
    """A ``rukh.infer.Player`` backed by a causal language model that writes SAN."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        stats: QwenStats | None = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        max_plies: int = DECODER_PLIES,
    ) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.stats = stats if stats is not None else QwenStats()
        self.max_new_tokens = max_new_tokens
        self.max_plies = max_plies
        self.white_elo = self.black_elo = 1800

    @property
    def limit(self) -> int:
        """Plies this player can carry; the decoder's number, so both play the same game."""
        return self.max_plies

    def start(self, white_elo: int, black_elo: int) -> None:
        self.white_elo, self.black_elo = white_elo, black_elo

    def observe(self, move: chess.Move) -> None:
        """Nothing to record: the board carries the history and the prompt is built from it."""

    def answer(self, board: chess.Board) -> str:
        """The raw continuation the model writes for this position."""
        import torch

        prompt = prompt_for(board, self.white_elo, self.black_elo)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        written = out[0, encoded["input_ids"].shape[1] :]
        return str(self.tokenizer.decode(written, skip_special_tokens=True))

    def propose(self, board: chess.Board) -> tuple[chess.Move | None, str]:
        """What the model wrote, parsed, with the outcome named. No mask, no rescue."""
        move, outcome = parse_san_answer(board, self.answer(board))
        self.stats.record(outcome)
        return move, outcome

    def choose(self, board: chess.Board) -> tuple[chess.Move | None, bool]:
        """The move to play, and whether the model's own answer was usable.

        The fallback is the first legal move in the position's own order, not a second sample:
        the model already answered, and asking again until it says something legal would measure
        a different model from the one the legality rate describes.
        """
        move, outcome = self.propose(board)
        if move is not None:
            return move, True
        legal = list(board.legal_moves)
        return (legal[0] if legal else None), False


def qwen_source(player: QwenPlayer) -> Any:
    """The ``MoveSource`` the puzzle suite expects, backed by the same player.

    Puzzles are scored on the move the model *chooses*, so the fallback applies here too and a
    position where it wrote nonsense counts as a failed puzzle rather than as a missing one.
    """

    def choose(board: chess.Board, history: list[int]) -> chess.Move | None:  # noqa: ARG001
        move, _ = player.choose(board)
        return move

    return choose
