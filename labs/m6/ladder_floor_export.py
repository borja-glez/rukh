"""The floor of the ladder, as a figure: four runs of the same model, two on a clock, two on nodes.

D-107 found that the same configuration with the same seed gave 1498 and 1558, because the
opponent plays on a clock and a clock is not the same twice. M6 measured it on purpose: the same
checkpoint, the same rungs, games and seed, twice with ``elo_move_time`` and twice with
``elo_nodes``. This collects the four ``results.json`` and writes the course figure's
``artifacts/web/ladder-floor.json``, with the spread between the two runs of each regime and the
regime that was chosen (D-135):

    uv run python labs/m6/ladder_floor_export.py --decision nodes
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from rukh.paths import resolve

RUNS = {
    "time-a": ("ladder-time-a", "time"),
    "time-b": ("ladder-time-b", "time"),
    "nodes-a": ("ladder-nodes-a", "nodes"),
    "nodes-b": ("ladder-nodes-b", "nodes"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--decision", choices=["time", "nodes"], required=True)
    parser.add_argument("--out", default="artifacts/web/ladder-floor.json")
    args = parser.parse_args()

    runs: list[dict[str, object]] = []
    model = None
    games = None
    for name, (stage, limit) in RUNS.items():
        path = resolve(f"artifacts/eval/{stage}/results.json")
        if not path.is_file():
            print(f"{name}: {path} missing, skipped")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        elo = result["elo"]
        model = model or result.get("checkpoint")
        games = games or elo.get("games")
        runs.append(
            {
                "name": name,
                "limit": limit,
                "elo": round(elo["elo"]),
                "lo": None if elo.get("ci_low") is None else round(elo["ci_low"]),
                "hi": None if elo.get("ci_high") is None else round(elo["ci_high"]),
                "date": result.get("date"),
            }
        )
        print(f"{name:<8} {limit:<6} {runs[-1]['elo']} ({runs[-1]['lo']}-{runs[-1]['hi']})")

    spread = {}
    for limit in ("time", "nodes"):
        pair = [r["elo"] for r in runs if r["limit"] == limit]
        if len(pair) == 2:
            spread[limit] = abs(int(pair[0]) - int(pair[1]))
            print(f"spread {limit}: {spread[limit]} Elo")

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "meta": {
                    "generated": datetime.now(UTC).isoformat(timespec="seconds"),
                    "model": model,
                    "games": games,
                },
                "runs": runs,
                "spread": spread,
                "decision": args.decision,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
