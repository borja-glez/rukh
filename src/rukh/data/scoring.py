"""One scoring convention for every multi-PV consumer (consolidation and DPO pairs).

``cp`` and ``mate`` in ``Lichess/chess-position-evaluations`` are from White's point of view.
A mate is worth ``MATE_SCORE - mate`` (so a mate in 1 beats a mate in 5) and a line is judged
from the side to move, which negates the score for Black.

Lines are ranked best first by: a mate *for the side to move*, then depth, then the score for
the side to move, then the move string as a tie-break. ``evals`` consolidates with the SQL
expressions below and ``pairs`` ranks with ``line_rank_key``; both orderings must stay
identical, which is why they live in one module.
"""

from __future__ import annotations

MATE_SCORE = 10_000
BEST_LINE_DOC = "deepest; mate for the mover first; then best cp for the side to move"

# SQL counterparts used by ``rukh.data.evals`` (``sign`` is ``mover_sign`` of the FEN turn).
SCORE_WHITE_SQL = f"""
CASE WHEN mate IS NOT NULL
     THEN CASE WHEN mate > 0 THEN {MATE_SCORE} - mate ELSE -{MATE_SCORE} - mate END
     ELSE cp END
"""
RANK_ORDER_SQL = (
    "(mate IS NOT NULL AND sign * mate > 0) DESC, depth DESC, sign * score_white DESC, move"
)


def mover_sign(turn: str) -> int:
    """``+1`` when White is to move, ``-1`` for Black."""
    return 1 if turn == "w" else -1


def score_white(cp: int | None, mate: int | None) -> int | None:
    """Centipawns from White's point of view; a mate in ``n`` is ``+/-(10000 - n)``."""
    if mate is not None:
        return MATE_SCORE - mate if mate > 0 else -MATE_SCORE - mate
    return None if cp is None else int(cp)


def score_mover(cp: int | None, mate: int | None, turn: str) -> int | None:
    """``score_white`` seen from the side to move: unchanged for White, negated for Black."""
    white = score_white(cp, mate)
    return None if white is None else mover_sign(turn) * white


def line_rank_key(
    cp: int | None, mate: int | None, depth: int | None, move: str, turn: str
) -> tuple[bool, int, int, str]:
    """Sort key (ascending) reproducing ``RANK_ORDER_SQL``: the best line sorts first."""
    sign = mover_sign(turn)
    white = score_white(cp, mate)
    if white is None:
        raise ValueError("a line without cp and without mate cannot be ranked")
    mates_for_mover = mate is not None and sign * mate > 0
    # DuckDB orders NULL depths last, so an unknown depth ranks below any known one.
    return (not mates_for_mover, -(depth if depth is not None else -1), -sign * white, move)
