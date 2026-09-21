# M2 labs

Scripts referenced by the M2 lesson (`rukh-lab`, `curso/m2/01-el-decoder`). Run them from the
repository root.

| Script | Needs | Lesson lab |
|---|---|---|
| `params.py` | nothing | 1 · contar los parámetros a mano |
| `causal_mask.py` | nothing | 1 · comprobar la causalidad |
| `attention_export.py` | a trained checkpoint | visualización · `artifacts/web/attention.json` |
| `replay_export.py` | an MLflow run and its `step-*.pt` files | visualización · `artifacts/web/training-replay.json` |

`params.py`, `causal_mask.py` and `attention_export.py` are copies of the code blocks in the
lesson: if you edit one, edit the other. `replay_export.py` is only described there, so this file
is its source of truth.

Both JSON outputs are copied into the course with `pnpm sync:data`.

## Starting point

M2 trains on `data/tokens/uci/` (M1's `rukh data tokenize --scheme uci --pack`) and evaluates
with `data/puzzles/` and `data/uci/`. `uv run rukh pull --module m2` brings the games, the
tokenizer, the puzzles, the evaluations and the two published decoders (`tiny`, `small`); the
packed tokens are rebuilt locally in minutes. The fourth part of the module (`medium-v4`, the
model M4 and M5 start from) needs the 44 months of Elite games: `rukh pull rukh-games-elite`, or
`rukh data elite --config configs/data/pipeline-elite44.yaml`. Every command, with its measured
duration, is in `docs/reproducir.md`.

A trained run lands in a stamped folder (`checkpoints/small-20260919-062911/`); the lessons and
the configs name it without the stamp (`checkpoints/small/best.pt`), and both spellings work.

## `replay_export.py`

One entry per **checkpoint**, not per MLflow logging step. The script reads the metric history of
a run, keeps the steps that still have a `step-*.pt` file next to them (plus the last logged step,
whatever happened to its weights) and writes `artifacts/web/training-replay.json` with the schema
`rukh-training-replay/1` documented in `rukh-lab/src/data/README.md`.

Why per checkpoint: the slider of the `TrainingReplay` island is advertised as walking the
checkpoints, and legality and Elo can only be measured for a step whose weights are on disk. With
the shipped configs (`log_every: 10`, `eval_every: 250`/`500`, `ckpt_every: 1000`) every checkpoint
step also has its three MLflow metrics, so the entries come out complete; the writer does not
promise it, though, and only `step` is guaranteed.

```bash
# the curve alone: seconds, needs nothing but the MLflow store
uv run python labs/m2/replay_export.py --run-name small-20260919-013000

# plus the unmasked argmax legality of every checkpoint (the >= 99 % bar of GOAL, D-026)
uv run python labs/m2/replay_export.py --run-name small-... --with-legality --positions 500

# plus a rough Elo per checkpoint: very slow, and 10 games per rung is a shape, not a number
uv run python labs/m2/replay_export.py --run-name small-... --with-elo --elo-games 10
```

| Field | When it is written |
|---|---|
| `step` | always |
| `train_loss`, `val_loss`, `val_top1` | when MLflow logged that metric at that exact step |
| `legality` | with `--with-legality` (or `--with-elo`), unmasked **argmax** rate |
| `elo` | with `--with-elo`, and only when Stockfish is available |

`--checkpoints DIR` overrides where the `step-*.pt` files are looked for; by default it is the
run's `out_dir` parameter joined with the run name, which is where `rukh train` wrote them.

`--with-elo` plays `--elo-games` games against each of the eight Stockfish rungs **for every
checkpoint**, so it is hours of games: it is there to draw the shape of the curve. The published
Elo, with its confidence interval, is the one `rukh eval --suite full` produces.
