# M5 labs

Scripts referenced by the M5 lessons (`rukh-lab`, `curso/m5/`). Run them from the repository
root.

| Script | Needs | Lesson lab |
|---|---|---|
| `games_needed_match.py` | nothing | 1 · how many games a head-to-head match needs |
| `pooled_match.py` | two `rukh eval match` reports, one per direction | 3 · pooling both directions of a match |
| `reward_hacking.py` | Stockfish | 4 and 8 · the reward-hacking gallery and the depth check |

## Starting point

Everything in M5 starts from `checkpoints/medium-v4/best.pt` and
`data/pairs/dpo-prompts.parquet`. If you skipped M2's fourth part or M1's pairs step:

```bash
uv run rukh pull --module m5          # medium-v4, the pairs, the puzzles and the three aligned models
```

The full command sequence of the module, with what each step takes, is
`docs/runbooks/alineamiento.md`; the per-module map is `docs/reproducir.md`.

## `games_needed_match.py`

Two ways to answer "is this model 50 Elo better?": subtract two ladder ratings, or play the two
models against each other. The script prints the games each one needs, and the ratio between
them (about 8.1). No data, no engine.

## `pooled_match.py`

`rukh eval match` plays `--a` against `--b` with mirrored colours, but the two sides are still
not symmetric (D-120: swapping them flipped a verdict). Run both directions, then:

```bash
uv run rukh eval match --a checkpoints/medium-v4-dpo-onpolicy/dpo.pt --b checkpoints/medium-v4/best.pt --games 400 --seed 7
uv run rukh eval match --a checkpoints/medium-v4/best.pt --b checkpoints/medium-v4-dpo-onpolicy/dpo.pt --games 400 --seed 7
uv run python labs/m5/pooled_match.py
```

It pools every pair of reports it finds under `artifacts/eval/`, prints the pooled Elo with its
interval in closed form, and, when three models form a triangle, how far the three edges are
from adding up, in sigmas.

## `reward_hacking.py`

Six rewards, five of them broken in exactly one term, scored on a handful of positions the
script builds with python-chess. `--depth` fixes the engine depth (D-118: no reward that reads
the engine is stable across depths, so two runs at different depths are not comparable).

```bash
uv run python labs/m5/reward_hacking.py --depth 10
```
