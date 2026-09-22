"""Pool the two directions of a match, and show how little the verdict deserves to be trusted.

A lab of M5. `rukh eval match` plays A against B; playing B against A is a different sample of the
same question, and M5 ran both for all three pairs because the first three matches all went to
whoever was passed as A and that had to be ruled out.

What came back was worth a lab of its own. For the pair `DPO on-policy` against `DPO off-policy`:

* on as A: +48 Elo, interval 19 to 80 -- **separated from zero**;
* off as A: -26 Elo for off, interval -57 to +3 -- **includes zero**.

Same two models, same four hundred games, same opening book, the sides swapped. The estimate moved
from +48 to +26 in favour of the same model, which is well inside the noise. The **verdict** flipped
from "this is a measurement" to "this is not". A binary read off the edge of an interval is far less
stable than the number it was read from, and the honest thing to do with two directions is to pool
them:

    uv run python labs/m5/pooled_match.py

It reads the `results.json` of every match under `artifacts/eval/` and, for each unordered pair,
reports the pooled score, the pooled Elo and its interval over all the games of both directions.
It also prints the triangle check: whether the three edges are consistent with a single strength
per model, which for M5's three models they are not.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Z = 1.959963985


def elo(score: float) -> float:
    """The same conversion `rukh.eval.match` uses, so the two agree by construction."""
    if score <= 0.0:
        return -1200.0
    if score >= 1.0:
        return 1200.0
    return 400.0 * math.log10(score / (1.0 - score))


def elo_interval(score: float, games: int, spread: float) -> tuple[float, float]:
    """Interval in Elo, from the standard error of the mean score.

    The derivative of the Elo curve at ``score`` turns an interval in score into one in Elo. Doing
    it this way rather than bootstrapping is deliberate: the two directions are separate files with
    separate game lists, and a closed form over the pooled mean needs no access to either.
    """
    if games <= 1 or not 0.0 < score < 1.0:
        return (float("nan"), float("nan"))
    standard_error = spread / math.sqrt(games)
    slope = 400.0 / (math.log(10.0) * score * (1.0 - score))
    half = Z * standard_error * slope
    return (elo(score) - half, elo(score) + half)


def load(root: Path) -> list[dict]:
    """Every match report under ``root``, with the fields this lab needs."""
    found = []
    for path in sorted(root.rglob("results.json")):
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if {"a", "b", "score", "played"} <= data.keys():
            found.append(data)
    return found


def pooled(entries: list[dict], first: str, second: str) -> dict | None:
    """Every game between these two, oriented so the score is ``first``'s, in one summary."""
    scores: list[float] = []
    directions = []
    for entry in entries:
        if {entry["a"], entry["b"]} != {first, second}:
            continue
        flip = entry["a"] != first
        games = [1.0 - game["score"] if flip else game["score"] for game in entry["played"]]
        scores.extend(games)
        directions.append(
            {
                "as": "B" if flip else "A",
                "games": len(games),
                "score": sum(games) / len(games),
                "elo": entry["elo"] * (-1 if flip else 1),
                "separated": entry["separated"],
            }
        )
    if not scores:
        return None
    mean = sum(scores) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores) / max(len(scores) - 1, 1)
    low, high = elo_interval(mean, len(scores), math.sqrt(variance))
    return {
        "first": first,
        "second": second,
        "games": len(scores),
        "score": mean,
        "elo": elo(mean),
        "low": low,
        "high": high,
        "separated": low > 0 or high < 0,
        "directions": directions,
        "standard_error": math.sqrt(variance) / math.sqrt(len(scores)),
        "slope": 400.0 / (math.log(10.0) * mean * (1.0 - mean)) if 0 < mean < 1 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/eval", help="Where the match reports live.")
    args = parser.parse_args()

    entries = load(Path(args.root))
    if not entries:
        raise SystemExit(f"no match reports under {args.root}")
    names = sorted({name for entry in entries for name in (entry["a"], entry["b"])})

    print("Cada par, en las dos direcciones y agrupado.\n")
    summaries: dict[frozenset[str], dict] = {}
    for first, second in itertools.combinations(names, 2):
        summary = pooled(entries, first, second)
        if summary is None:
            continue
        summaries[frozenset((first, second))] = summary
        print(f"== {first} contra {second} ==")
        for direction in summary["directions"]:
            mark = "separado" if direction["separated"] else "incluye el cero"
            print(
                f"   como {direction['as']}  {direction['games']:>4} partidas  "
                f"{direction['score']:.4f}  {direction['elo']:+6.0f} Elo   {mark}"
            )
        verdict = "separado" if summary["separated"] else "incluye el cero"
        print(
            f"   agrupado {summary['games']:>4} partidas  {summary['score']:.4f}  "
            f"{summary['elo']:+6.0f} Elo   IC {summary['low']:+.0f} a {summary['high']:+.0f}"
            f"   {verdict}\n"
        )

    if len(names) < 3:
        return
    print(
        "El triángulo: si la fuerza fuera un solo número por modelo, las tres aristas cerrarían.\n"
    )
    for trio in itertools.combinations(names, 3):
        edges = [summaries.get(frozenset(pair)) for pair in itertools.combinations(trio, 2)]
        if any(edge is None for edge in edges):
            continue
        # Orient every edge as "later name minus earlier name" so the three add up to zero when
        # a single strength per model explains them.
        values, errors = [], []
        for edge, pair in zip(edges, itertools.combinations(trio, 2), strict=True):
            sign = 1.0 if edge["first"] == pair[0] else -1.0
            values.append(sign * edge["elo"])
            errors.append(edge["standard_error"] * edge["slope"])
        # (a-b) + (b-c) - (a-c) is zero for any consistent set of strengths.
        residual = values[0] + values[2] - values[1]
        spread = math.sqrt(sum(error**2 for error in errors))
        print(f"   {' / '.join(trio)}")
        print(
            f"   residuo {residual:+.0f} Elo, error típico {spread:.0f}  "
            f"-> {abs(residual) / spread:.1f} sigma"
        )
        print(
            "   "
            + (
                "consistente con un solo número por modelo."
                if abs(residual) < 2 * spread
                else "NO es consistente: el emparejamiento importa, no solo la fuerza."
            )
        )


if __name__ == "__main__":
    main()
