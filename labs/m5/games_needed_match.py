"""How many games a head-to-head match needs, and what a match of N games can see.

The first thing that runs in M5. `docs/acceptance.md` asks for **+50 Elo with an
interval** over the base, and there are two ways to measure that:

* run the Stockfish ladder twice and subtract the two absolute ratings;
* play the two models against each other and read the difference directly.

They are not equally good and the difference is not a matter of taste:

    uv run python labs/m5/games_needed_match.py

The second one is about eight times cheaper, and it also drops a source of noise the first one
cannot avoid -- P4 measured that two identical ladder runs of the same model give 1498 and 1558
(D-107), because Stockfish plays on a clock and `UCI_LimitStrength` randomises on purpose. In a
match both models play the same game, so there is no third party whose mood has to be averaged.

The rule this leaves: **to measure a difference, measure the difference.**
"""

import argparse
import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Z = 1.959963985  # two-sided 95 %

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--edges", default="10,25,35,50,75,100", help="Elo edges to price, in Elo.")
parser.add_argument("--ladder-base", type=float, default=0.45, help="Score the ladder sits at.")
args = parser.parse_args()


def score_for(elo: float) -> float:
    """The expected score of a player ``elo`` points stronger than its opponent."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def match_games(elo: float) -> int:
    """Games of a head-to-head match needed to separate this edge from zero."""
    p = score_for(abs(elo))
    return math.ceil(Z * Z * p * (1 - p) / (p - 0.5) ** 2)


def ladder_games(elo: float, base: float) -> int:
    """Games **per side** needed to separate two ladder estimates this far apart.

    The ladder measures a score against fixed rungs, so an edge in Elo has to be converted into
    the score it moves *there*, which depends on where on the curve the model sits.
    """
    slope = math.log(10) / 400 * base * (1 - base)  # d(score) / d(Elo)
    delta = abs(elo) * slope
    spread = 2 * math.sqrt(base * (1 - base))
    return math.ceil((Z * spread / delta) ** 2)


print(f"{'ventaja':>9}  {'tasa h2h':>9}  {'partidas h2h':>13}  {'escalera (por lado)':>20}  ratio")
for text in args.edges.split(","):
    elo = float(text)
    match, ladder = match_games(elo), ladder_games(elo, args.ladder_base)
    print(
        f"{elo:>+8.0f}  {score_for(elo):>9.4f}  {match:>13,}  {ladder:>20,}  "
        f"{2 * ladder / match:>4.1f}x"
    )

print(
    "\nY al reves: con N partidas de enfrentamiento, que ventaja se puede ver.\n"
    f"{'partidas':>9}  {'ventaja minima':>15}"
)
for games in (50, 100, 200, 400, 800):
    # p - z sqrt(p(1-p)/n) > 0.5, resuelto para p y pasado a Elo.
    p = 0.5 + Z / (2 * math.sqrt(games)) * math.sqrt(1 + Z * Z / games) / (1 + Z * Z / games)
    print(f"{games:>9,}  {400 * math.log10(p / (1 - p)):>14.0f} Elo")

print(
    "\nun negativo con pocas partidas no dice que no haya efecto:\n"
    "dice que el instrumento no lo habria visto"
)
