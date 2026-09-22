"""Measure the ladder against itself: every rung plays the anchor, engine against engine.

A lab of M6, and the method of D-070 turned into a script. The four ``Skill Level`` rungs of the
Elo ladder were once labelled 800/950/1100/1250 by assumption and measured 1381/1467/1589/1678;
the labels of a ladder are hypotheses until the rungs have played each other. This plays each
rung against ``uci-1320`` (the anchor, the floor of ``UCI_Elo``) with colours alternated and a
seeded random opening per pair, and prints the rating that explains the score, with its
interval, next to the label the ladder carries.

It takes the same limit the harness takes -- a clock or a node budget -- so the ladder can be
calibrated for the regime it is going to be used in:

    uv run python labs/m6/ladder_check.py --games 40                 # 0.1 s per move, as P2-P5
    uv run python labs/m6/ladder_check.py --games 40 --nodes 200000  # the node budget of M6

The control is built in: ``uci-1500`` against ``uci-1320`` should come out near +180. If it
does not, the procedure is biased and no other row of the table means anything.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import chess
import numpy as np

from rukh.eval.elo import DEFAULT_RUNGS, EloRung
from rukh.eval.match import opening_book
from rukh.infer.game import StockfishOpponent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANCHOR = DEFAULT_RUNGS[0]
MAX_PLIES = 300


def elo_of(score: float) -> float:
    """The rating difference that explains a score, clipped away from 0 and 1."""
    p = min(max(score, 1e-3), 1 - 1e-3)
    return 400.0 * math.log10(p / (1.0 - p))


def play(white: StockfishOpponent, black: StockfishOpponent, opening: list[str]) -> float:
    """One game from ``opening``; the score from White's point of view."""
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        mover = white if board.turn == chess.WHITE else black
        board.push(mover.choose(board))
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        return {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
    return 0.5  # out of plies: a draw, the same adjudication the harness would not accept but fair


def measure(
    rung: EloRung, games: int, seed: int, move_time: float, nodes: int | None
) -> tuple[float, float, float, float]:
    """Score of ``rung`` against the anchor, its Elo over the anchor and a bootstrap interval."""
    book = opening_book(games // 2, plies=6, seed=seed)
    scores: list[float] = []
    with (
        StockfishOpponent(
            elo=rung.uci_elo or 1320, skill=rung.skill, move_time=move_time, nodes=nodes
        ) as candidate,
        StockfishOpponent(elo=ANCHOR.uci_elo or 1320, move_time=move_time, nodes=nodes) as anchor,
    ):
        for opening in book:
            scores.append(play(candidate, anchor, opening))  # rung as White
            scores.append(1.0 - play(anchor, candidate, opening))  # rung as Black
    draws = np.asarray(scores)
    rng = np.random.default_rng(seed)
    boots = [
        elo_of(float(rng.choice(draws, size=len(draws), replace=True).mean())) for _ in range(500)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(draws.mean()), elo_of(float(draws.mean())), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", type=int, default=40, help="Games per rung (even).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--move-time", type=float, default=0.1)
    parser.add_argument("--nodes", type=int, default=None, help="Node budget instead of the clock.")
    args = parser.parse_args()
    if args.games % 2:
        raise SystemExit("--games must be even: every opening is played with both colours")
    regime = f"{args.nodes} nodes" if args.nodes else f"{args.move_time:g} s per move"
    print(f"anchor {ANCHOR.name} ({ANCHOR.elo}) · {args.games} games per rung · {regime}\n")
    print(f"{'rung':<10}{'label':>7}{'score':>8}{'measured':>10}{'IC 95 %':>16}{'delta':>8}")
    started = time.perf_counter()
    for rung in DEFAULT_RUNGS[1:]:
        score, over, lo, hi = measure(rung, args.games, args.seed, args.move_time, args.nodes)
        measured = ANCHOR.elo + over
        print(
            f"{rung.name:<10}{rung.elo:>7}{score:>8.3f}{measured:>10.0f}"
            f"{ANCHOR.elo + lo:>8.0f}-{ANCHOR.elo + hi:<7.0f}{measured - rung.elo:>+8.0f}"
        )
    print(f"\n{time.perf_counter() - started:.0f} s")
    print("control: uci-1500 should read about +180 over the anchor; if it does not, stop here.")


if __name__ == "__main__":
    main()
