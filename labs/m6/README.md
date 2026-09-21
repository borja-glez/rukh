# M6 labs

Scripts referenced by the M6 lessons (`rukh-lab`, `curso/m6/`). Run them from the repository
root. Nothing here trains: the module measures, exports and publishes what M1-M5 left.

| Script | Needs | Lesson lab |
|---|---|---|
| `ladder_check.py` | Stockfish | 1 · every rung of the ladder against the anchor, engine against engine |
| `ladder_floor_export.py` | the four `ladder-*` results | 1 · the floor of the instrument: two runs on a clock, two on nodes |
| `parity_cost.py` | the ONNX export of a model, Stockfish | 3 · what fp16 and int8 cost in Elo, on the same ladder |
| `puzzles_export.py` | `data/puzzles/` | 6 · the 150 puzzles the demo plays live |

## Starting point

M6 measures everything the course published. If you skipped any module:

```bash
uv run rukh pull --module m6          # every model, the games, the puzzles and the positions of the suite
uv run python scripts/get_stockfish.py
```

The per-module map with every command and what it took is `docs/reproducir.md` (section M6).

## `ladder_check.py`

The rungs of the Elo ladder are hypotheses until they have played each other (D-070: four rungs
were labelled 800-1250 by assumption and measured 1381-1678). Each rung plays the anchor
(`uci-1320`) with mirrored colours and a seeded random opening; the script prints the rating
that explains the score next to the label the ladder carries. `--nodes 200000` measures the
regime M6 chose; without it, the clock of P2-P5. The control is `uci-1500`: about +180 over the
anchor, or stop.

## `ladder_floor_export.py`

Collects `artifacts/eval/ladder-{time,nodes}-{a,b}/results.json` (the same model, rungs, games
and seed, twice per regime) into `artifacts/web/ladder-floor.json` with the spread of each
pair and the decision, for the course figure `LadderFloor`.

## `parity_cost.py`

Parity (M2) says how often an export picks the checkpoint's move; it cannot say what the other
positions cost. This plays `model.onnx`, `model-fp16.onnx` and `model-int8.onnx` on the ladder
through the same `DecoderPlayer` the checkpoint used (`rukh.infer.onnx_player.OnnxDecoder`
stands in for the model), and writes one interval per precision next to its parity. ONNX Runtime
runs on the CPU, so use a node budget: on a clock, a slower player changes the engine's side of
the game too.

## `puzzles_export.py`

Fifty puzzles per difficulty band, drawn with a fixed seed from the test split, with the real
game prefix and the players' ratings so the demo prompts the model exactly as the harness does.
