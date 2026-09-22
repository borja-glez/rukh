"""The gallery: five reward designs that are wrong, and what each one actually breaks.

A lab of M5. A verifiable reward is written down by a person, which means it is exactly
specifiable and therefore exactly gameable. The usual way this gets taught is a story about a boat
in a video game spinning in circles, and the trouble with a story is that you cannot check it.

So this lab does not tell you about reward hacking. It takes the reward `rukh.train.rewards`
ships, breaks it in five ways that each look reasonable on the page, and measures all six against
the same candidate group in the same positions.

    uv run python labs/m5/reward_hacking.py

**The first version of this lab asked the wrong question and the run said so.** It compared which
move each reward crowned, and all six crowned the same move in every position -- which is obvious
in hindsight, since every one of them is monotone in the engine's score. The top of the ranking is
not where reward hacking lives.

What it measures instead is what the optimiser actually consumes. GRPO does not see the reward: it
sees ``(r - mean) / std`` over the group. So the three columns are

* **corona** -- the move on top, kept because the negative result is part of the lesson;
* **cabeza** -- the advantage gap between the best candidate and the second best, which is how
  much signal is left to tell a good move from a decent one;
* **peor** -- the share of the group's whole advantage mass absorbed by its single worst
  candidate, which is how much of the step one blunder gets to decide.

And then the inversions: pairs of candidates that the sound reward and a broken one put in the
opposite order. An inversion is the hack in its smallest possible form -- one position, two moves,
a reason you can read.

The candidate group is built the way GRPO's actually is, illegal proposals included, because a
gate with nothing to gate proves nothing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import chess
import chess.engine

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rukh.data.scoring import MATE_SCORE  # noqa: E402
from rukh.engine import find_stockfish  # noqa: E402
from rukh.train.grpo import group_advantages  # noqa: E402
from rukh.train.rewards import (  # noqa: E402
    RewardWeights,
    illegal_value,
    mate_bonus,
    normalised_delta_cp,
    repetition_penalty,
)

WEIGHTS = RewardWeights()
CP_SCALE = WEIGHTS.cp_scale


@dataclass(frozen=True)
class Case:
    """A position, the moves that led to it, and the mistake it is here to expose."""

    name: str
    about: str
    fen: str = ""
    moves: tuple[str, ...] = ()


CASES = (
    Case(
        name="una pifia a mano",
        about="an ordinary position with a 900-centipawn blunder available",
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    ),
    Case(
        name="hay mate",
        about="a back-rank mate in one, beside moves that merely keep winning",
        fen="6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    ),
    Case(
        name="ganando y repitiendo",
        about="White is a queen up with no mate in one, and the repetition is on the board",
        fen="6k1/5pp1/7p/8/8/8/5PPP/3Q2K1 w - - 0 1",
        moves=("d1d2", "g8h7", "d2d1", "h7g8"),
    ),
)


def illegal_candidates(board: chess.Board, count: int = 2) -> list[str]:
    """Moves a model could plausibly propose here and that cannot be played.

    Generated rather than written down, because the first version of this lab hard-coded two per
    position and both turned out to be **legal** -- ``Ra1a7`` and ``Qd1d8`` are perfectly ordinary
    moves. The gallery then reported inversions that were an artefact of the labelling. Asking
    ``python-chess`` is the only way to be sure.

    They start from a square that really does hold one of the mover's pieces, so they look like
    proposals rather than noise: that is the shape of the illegal moves a decoder actually emits.
    """
    legal = {move.uci() for move in board.legal_moves}
    found: list[str] = []
    for from_square in chess.scan_forward(board.occupied_co[board.turn]):
        for to_square in chess.SQUARES:
            if from_square == to_square:
                continue
            uci = chess.square_name(from_square) + chess.square_name(to_square)
            if uci not in legal and board.piece_at(to_square) is None:
                found.append(uci)
                break
        if len(found) >= count:
            break
    return found[:count]


def board_of(case: Case) -> chess.Board:
    board = chess.Board(case.fen) if case.fen else chess.Board()
    for uci in case.moves:
        board.push_uci(uci)
    return board


def analyse(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> dict[str, int]:
    """Centipawns for the mover, for every legal move."""
    scores: dict[str, int] = {}
    for move in board.legal_moves:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        scores[move.uci()] = int(info["score"].pov(not board.turn).score(mate_score=MATE_SCORE))
        board.pop()
    return scores


def candidate_group(board: chess.Board, scores: dict[str, int]) -> list[tuple[str, int | None]]:
    """A group shaped like one GRPO would draw: good, middling, awful, repeating and illegal.

    Hand-picked rather than sampled, because the point is to give every failure mode somewhere to
    go. A real group is drawn from the policy and is usually far less interesting, which is its own
    reason these bugs survive a smoke test.
    """
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    picked: dict[str, int | None] = {}
    for uci, cp in (ranked[0], ranked[1], ranked[len(ranked) // 2], ranked[-1]):
        picked[uci] = cp
    for uci, cp in ranked:
        if repetition_penalty(board, chess.Move.from_uci(uci)) > 0:
            picked[uci] = cp
            break
    for uci in illegal_candidates(board):
        picked.setdefault(uci, None)
    return list(picked.items())


# --- the six rewards --------------------------------------------------------------------------
#
# Every variant is the sound reward with **exactly one** term replaced. That constraint is what
# makes the inversion column readable: the first version of this lab wrote each broken reward from
# scratch, so four of them silently dropped the repetition term too and every one of them reported
# the same inversion. A gallery whose exhibits differ in more than one way explains nothing.
#
# ``cp`` is ``None`` for a move that cannot be played, which is the moment each design has to say
# what it believes. ``span`` is the group's own spread, which only the self-scaling one looks at.


def _quality(cp: int, best: int) -> float:
    return normalised_delta_cp(cp, best, CP_SCALE)


def _mate(board: chess.Board, move: chess.Move) -> float:
    return mate_bonus(board, move)


def _repetition(board: chess.Board, move: chess.Move) -> float:
    return repetition_penalty(board, move)


def _assemble(quality: float, mate: float, repetition: float) -> float:
    return WEIGHTS.quality * quality + WEIGHTS.mate * mate - WEIGHTS.repetition * repetition


def sound(board: chess.Board, move: chess.Move, cp: int | None, best: int, span: float) -> float:
    """What `rukh.train.rewards.reward` computes: gated, capped, flat mate, repetition paid."""
    if cp is None:
        return illegal_value(WEIGHTS)
    return _assemble(_quality(cp, best), _mate(board, move), _repetition(board, move))


def uncapped(board: chess.Board, move: chess.Move, cp: int | None, best: int, span: float) -> float:
    """**Mistake 1: no floor under the quality term.** Only the quality term changes.

    ``(cp - best) / cp_scale`` reads like the same thing as the clamped version and is not: it has
    no bottom, so one catastrophic candidate sets the group's standard deviation by itself and
    every other advantage is divided by it. Watch the ``peor`` column, not the crown.
    """
    if cp is None:
        return illegal_value(WEIGHTS)
    return _assemble((cp - best) / CP_SCALE, _mate(board, move), _repetition(board, move))


def legality_as_a_bonus(
    board: chess.Board, move: chess.Move, cp: int | None, best: int, span: float
) -> float:
    """**Mistake 2: legality is a term in the sum instead of a gate.** Only the gate changes.

    ``+2 when legal`` reads as a strong preference and is in fact a *price*, which any other term
    worth more than two can pay. An illegal move has no engine score, so this design does the
    generous thing and treats the missing quality as no loss at all -- which is precisely how the
    bug gets in.
    """
    legal = move in board.legal_moves
    quality = _quality(cp, best) if cp is not None else 0.0
    return (2.0 if legal else 0.0) + _assemble(
        quality, _mate(board, move) if legal else 0.0, _repetition(board, move) if legal else 0.0
    )


def mate_by_engine_score(
    board: chess.Board, move: chess.Move, cp: int | None, best: int, span: float
) -> float:
    """**Mistake 3: paying for the engine's mate score instead of for the mate.** Mate term only.

    ``MATE_SCORE - distance`` is right there in the data and it is tempting. Two things go wrong
    and both are visible below: the term stops being bounded, so a single mating candidate takes
    the whole group's advantage mass; and the number it pays **depends on the search depth**,
    because a deeper engine reports a shorter mate for the same move. A reward that changes when
    the analysis budget changes is a reward that cannot be reproduced -- which is the same trap
    P4 fell into with a time-limited engine (D-107), arriving from a different direction.
    """
    if cp is None:
        return illegal_value(WEIGHTS)
    mate = (abs(cp) / 1000.0) if abs(cp) >= MATE_SCORE - 100 else 0.0
    return _assemble(_quality(cp, best), mate, _repetition(board, move))


def no_repetition_penalty(
    board: chess.Board, move: chess.Move, cp: int | None, best: int, span: float
) -> float:
    """**Mistake 4: rewarding the position instead of the progress.** Repetition term only.

    Drop it and a winning model discovers something true: repeating costs nothing. The evaluation
    after the repetition is the evaluation before it, so the reward is the same, and shuffling is a
    strictly safe way to keep collecting it. Nothing in the numbers is wrong, which is what makes
    this the most dangerous of the five.
    """
    if cp is None:
        return illegal_value(WEIGHTS)
    return _assemble(_quality(cp, best), _mate(board, move), 0.0)


def self_scaling(
    board: chess.Board, move: chess.Move, cp: int | None, best: int, span: float
) -> float:
    """**Mistake 5: a reward that rescales itself to whatever it is looking at.** Quality only.

    Normalising by the spread of *this* group instead of by a fixed constant makes every group look
    equally informative, so eight excellent moves and eight blunders produce advantages of the same
    size and the model is told, with equal confidence, to prefer the least bad blunder. It also
    stops the number being comparable between two runs, so nothing can tell a run it has stopped
    improving.
    """
    if cp is None:
        return illegal_value(WEIGHTS)
    return _assemble(
        1.0 - (best - cp) / max(span, 1.0), _mate(board, move), _repetition(board, move)
    )


REWARDS = {
    "sana": sound,
    "sin suelo": uncapped,
    "legalidad como bonus": legality_as_a_bonus,
    "mate por marcador": mate_by_engine_score,
    "sin penalizar repetir": no_repetition_penalty,
    "reescalada sola": self_scaling,
}


def profile(
    board: chess.Board, group: list[tuple[str, int | None]], function: object, best: int
) -> tuple[str, float, float, dict[str, float]]:
    """Crowned move, head gap, worst-candidate dominance, and the advantage of each candidate."""
    scored = [cp for _uci, cp in group if cp is not None]
    span = float(best - min(scored)) if scored else 1.0
    paid = {
        uci: float(function(board, chess.Move.from_uci(uci), cp, best, span))  # type: ignore[operator]
        for uci, cp in group
    }
    order = sorted(paid, key=lambda uci: paid[uci], reverse=True)
    advantages = dict(
        zip(paid, group_advantages([paid[uci] for uci in paid]).tolist(), strict=True)
    )
    mass = sum(abs(value) for value in advantages.values()) or 1.0
    head = advantages[order[0]] - advantages[order[1]]
    worst = min(advantages.values())
    return order[0], head, abs(worst) / mass, advantages


def inversions(reference: dict[str, float], other: dict[str, float]) -> list[tuple[str, str]]:
    """Pairs the two rewards order the opposite way round. Ties on either side do not count."""
    names = list(reference)
    found = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            sound_says = reference[first] - reference[second]
            other_says = other[first] - other[second]
            if sound_says * other_says < 0:
                found.append((first, second) if sound_says > 0 else (second, first))
    return found


DEPTH_CASE = Case(
    name="rey y torre contra rey",
    about="every move wins; only the search depth decides which ones the engine calls a mate",
    fen="7k/8/6K1/8/8/8/8/R7 w - - 0 1",
)
DEPTH_MOVE = "g6f6"


def depth_check(binary: object, depths: tuple[int, ...]) -> None:
    """The same move, four search depths: how much of each reward is really about the position.

    This is the half of mistake 3 that no ranking can show, and the answer turned out to be worse
    than the one the lab was written to give. ``Kf6`` in a king-and-rook endgame is scored 597 at
    depth 4 and 9997 at depth 6 -- not because the move changed, but because the engine found the
    mate. So the reward of a fixed move moves with the analysis budget, and it moves for **every**
    design that reads the engine's number, including the sound one.

    The only term that does not move is the flat mate bonus, because it asks the board whether
    this move is checkmate rather than asking the engine how it feels about the position. Which
    gives the rule the lesson keeps: **pin the depth, record it in the run, and never compare two
    runs that used different ones** -- the same conclusion P4 reached about the clock (D-107).
    """
    board = board_of(DEPTH_CASE)
    move = chess.Move.from_uci(DEPTH_MOVE)
    print("== el mismo movimiento a cuatro profundidades ==")
    print(f"   {DEPTH_CASE.name}, {DEPTH_MOVE}")
    print(f"   {DEPTH_CASE.about}")
    header = "   " + f"{'recompensa':<22}" + "".join(f"{d:>9}" for d in depths) + f"{'rango':>9}"
    print(header)
    values: dict[str, list[float]] = {label: [] for label in REWARDS}
    engine_says: list[int] = []
    with chess.engine.SimpleEngine.popen_uci(str(binary)) as engine:  # type: ignore[arg-type]
        for depth in depths:
            scores = analyse(engine, board, depth)
            best = max(scores.values())
            group = candidate_group(board, scores)
            span = float(best - min(cp for _uci, cp in group if cp is not None))
            engine_says.append(scores[DEPTH_MOVE])
            for label, function in REWARDS.items():
                values[label].append(
                    float(function(board, move, scores[DEPTH_MOVE], best, span))  # type: ignore[operator]
                )
    shown = "".join(f"{cp:>9}" for cp in engine_says)
    print(f"   {'(el motor dice)':<22}{shown}")
    for label, series in values.items():
        spread = max(series) - min(series)
        row = "".join(f"{value:>9.3f}" for value in series)
        print(f"   {label:<22}{row}{spread:>9.3f}")
    print()
    print(
        "   Ninguna es estable, y el rango dice cuanto: la sana se mueve 0.99, que es todo su\n"
        "   intervalo legal; la que no tiene suelo se mueve 47.0, cuarenta y siete veces ese\n"
        "   intervalo. La unica pieza que no se mueve es el bono de mate plano, porque le\n"
        "   pregunta al tablero y no al motor.\n"
    )
    print(
        "   Y hay una fila que se mueve sin que el motor se mueva: la reescalada sola pasa de\n"
        "   1.000 a 0.667 entre profundidad 10 y 14 con el mismo 9998 delante. No cambio la\n"
        "   jugada ni la posicion ni la evaluacion: cambio el grupo, que es lo unico que esa\n"
        "   recompensa mira. De ahi la regla: fija la profundidad, escribela en el run y no\n"
        "   compares dos runs con profundidades distintas.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=12, help="Engine depth per legal move.")
    args = parser.parse_args()

    binary = find_stockfish()
    if binary is None:
        raise SystemExit("Stockfish not found; set RUKH_STOCKFISH")

    print("La galeria: el mismo grupo de candidatas, seis recompensas, lo que cada una rompe.\n")
    with chess.engine.SimpleEngine.popen_uci(str(binary)) as engine:
        for case in CASES:
            board = board_of(case)
            scores = analyse(engine, board, args.depth)
            best = max(scores.values())
            group = candidate_group(board, scores)
            shown = ", ".join(f"{uci}({'ilegal' if cp is None else cp})" for uci, cp in group)
            print(f"== {case.name} ==")
            print(f"   {case.about}")
            print(f"   grupo: {shown}")
            print(f"   {'recompensa':<22} {'corona':<7} {'cabeza':>8} {'peor':>7}   inversiones")
            reference: dict[str, float] = {}
            for label, function in REWARDS.items():
                crowned, head, worst, advantages = profile(board, group, function, best)
                if not reference:
                    reference = advantages
                    flips = ""
                else:
                    pairs = inversions(reference, advantages)
                    listed = ", ".join(f"{a}>{b}" for a, b in pairs[:3])
                    flips = (listed + (" ..." if len(pairs) > 3 else "")) if pairs else "-"
                print(f"   {label:<22} {crowned:<7} {head:>8.3f} {worst:>6.1%}   {flips}")
            print()

    depth_check(binary, (4, 6, 10, 14))
    print(
        "Lo que mide, en una linea: la corona casi nunca cambia -- todas son monotonas en la\n"
        "evaluacion -- y el dano esta en la forma del grupo, no en su cima."
    )


if __name__ == "__main__":
    main()
