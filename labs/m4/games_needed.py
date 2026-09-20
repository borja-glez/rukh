"""How many games each pair of conditions would need before its intervals stop overlapping.

Lab 6 of M4. The sweep of this milestone leaves `1500 < 2000 < 2400` unproven, and there are two
very different reasons a comparison can come out unproven: too few games, or no difference to
find. Telling them apart is arithmetic, not opinion.

The ladder plays `n` games per condition and scores them, so the score is a proportion and its
standard error is `sqrt(p (1 - p) / n)`. Two 95 % intervals stop touching when the gap between
the scores is wider than `1.96 * (se1 + se2)`, which for two proportions near a half means

    n  >  3.84 / (delta p)^2

per condition. Put the measured gap in and the answer comes out in games. A pair that needs two
hundred is a pair worth re-running; a pair that needs twenty thousand is a pair where the honest
conclusion is that the difference is not there, and no amount of machine time will change it.

Run from the repository root, on the sweep's own file:

    uv run python labs/m4/games_needed.py
    uv run python labs/m4/games_needed.py --only 1500,2000,2400

``--only`` keeps just the conditions it names, in the order they are given, which is how the
acceptance criterion of this project is read: it asks about `<w1500>`, `<w2000>` and `<w2400>`
and about no other pair.
"""

import argparse
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Z = 1.959963985  # two-sided 95 %

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", default="artifacts/eval/medium-elo-elo-sweep/results.json")
parser.add_argument("--games", type=int, default=160, help="Games per condition already played.")
parser.add_argument("--only", default="", help="Comma-separated headers to keep, e.g. 1500,2000.")
args = parser.parse_args()

rows = json.loads(Path(args.results).read_text(encoding="utf-8"))["rows"]
if args.only:
    wanted = [int(value) for value in args.only.split(",")]
    by_header = {int(row["header_elo"]): row for row in rows}
    missing = [elo for elo in wanted if elo not in by_header]
    if missing:
        raise SystemExit(f"{args.results} has no condition for {missing}")
    rows = [by_header[elo] for elo in wanted]
if len(rows) < 2:
    raise SystemExit(f"{args.results} has {len(rows)} condition(s); a comparison needs two")


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 1e-12) / n)


def needed(p1: float, p2: float) -> float:
    """Games per condition for the two 95 % intervals to stop touching."""
    gap = abs(p2 - p1)
    if gap <= 0:
        return math.inf
    spread = math.sqrt(p1 * (1 - p1)) + math.sqrt(p2 * (1 - p2))
    return (Z * spread / gap) ** 2


print(f"{args.results}: {len(rows)} conditions, {args.games} games each\n")
print("  pair                 score      gap   separated now   games needed each")
for low, high in zip(rows, rows[1:], strict=False):
    p1, p2 = low["score"], high["score"]
    gap = p2 - p1
    apart = (p1 + Z * se(p1, args.games)) < (p2 - Z * se(p2, args.games))
    n = needed(p1, p2)
    n_text = "never" if not math.isfinite(n) else f"{math.ceil(n):,}"
    print(
        f"  <w{low['header_elo']}> -> <w{high['header_elo']}>  "
        f"{p1:.3f}->{p2:.3f}  {gap:+.3f}   {'yes' if apart else 'no ':<13}   {n_text}"
    )

ends = (rows[0], rows[-1])
p1, p2 = ends[0]["score"], ends[1]["score"]
apart = (p1 + Z * se(p1, args.games)) < (p2 - Z * se(p2, args.games))
print(
    f"\n  ends <w{ends[0]['header_elo']}> -> <w{ends[1]['header_elo']}>: "
    f"{p1:.3f} -> {p2:.3f}, separated now: {'yes' if apart else 'no'}, "
    f"needs {math.ceil(needed(p1, p2)):,} games each"
)
print(
    "\na pair that needs a few hundred games is a measurement problem;\n"
    "a pair that needs tens of thousands is a difference that is not there"
)
