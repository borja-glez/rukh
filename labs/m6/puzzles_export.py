"""Export a small, seeded set of test puzzles for the demo's live puzzle mode.

A lab of M6. The demo cannot read a parquet and should not download 22 MB to show fifty puzzles,
so this writes `artifacts/web/puzzles.json`: 50 puzzles per difficulty band drawn from the
`test` split of `chorcat/rukh-puzzles-split` with the same loader the harness uses
(`rukh.eval.puzzles.load_puzzles`), so what the browser attempts is exactly what the table
measured -- the real game prefix, the players' own ratings in the header, and the whole line as
the criterion (D-053). The file is copied into `rukh-web/public/puzzles.json` by hand (it is
versioned there: the demo must not depend on this repository being around).

    uv run python labs/m6/puzzles_export.py            # 50 per band, seed 6
    uv run python labs/m6/puzzles_export.py --per-band 20 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from rukh.eval.puzzles import load_puzzles
from rukh.paths import resolve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCHEMA = "rukh-puzzles/1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--puzzles", default="data/puzzles/puzzles.parquet")
    parser.add_argument("--per-band", type=int, default=50)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="artifacts/web/puzzles.json")
    args = parser.parse_args()

    items = load_puzzles(resolve(args.puzzles), args.per_band, seed=args.seed, split=args.split)
    without_prefix = sum(1 for item in items if not item.prefix)
    if without_prefix:
        raise SystemExit(
            f"{without_prefix} puzzles carry no game prefix; the demo prompts with the real game "
            "(D-053), so the parquet must be the `with_games` one"
        )
    bands: dict[str, list[dict[str, object]]] = {}
    for item in items:
        bands.setdefault(item.band, []).append(
            {
                "id": item.puzzle_id,
                "fen": item.fen,
                "moves": item.moves,
                "rating": item.rating,
                "prefix": item.prefix,
                "whiteElo": item.white_elo,
                "blackElo": item.black_elo,
            }
        )
    payload = {
        "schema": SCHEMA,
        "source": f"chorcat/rukh-puzzles-split · split {args.split}",
        "seed": args.seed,
        "perBand": args.per_band,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "bands": bands,
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", "utf-8")
    print(f"{out} · {sum(len(v) for v in bands.values())} puzzles · {out.stat().st_size:,} bytes")
    for band, rows in sorted(bands.items()):
        ratings = [row["rating"] for row in rows]
        print(f"  {band:<10} {len(rows):>3} puzzles · rating {min(ratings)}-{max(ratings)}")


if __name__ == "__main__":
    main()
