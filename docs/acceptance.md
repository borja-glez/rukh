# Acceptance bars

The numbers this project wrote down **before** anything was trained, and what each one turned out
to be. The code cites this file: `rukh.eval.suite.GOAL_LEGALITY`, `rukh.eval.encoder.GOAL_MARGIN`
and `GOAL_VALUE_CORRELATION` are these bars, the encoder suite checks its two against every run,
and the model cards publish the verdict table whether or not the answer is flattering.

A bar that moves after the measurement is not a bar, so none of these were rewritten. Where one was
missed the row says so, with what was measured instead and the decision that says what the project
did about it.

| Milestone | Bar                                                             | Measured                | Verdict       |
| --------- | --------------------------------------------------------------- | ----------------------- | ------------- |
| P2        | Legal move ≥ 99 % of the time, argmax, no mask                    | 99.4 % (`small-greedy`) | met           |
| P2        | `small` reaches ≥ 1200 Elo, interval included                     | 1355 (CI 1297-1420)     | met           |
| P3        | Blunder F1 ≥ the material heuristic + 5 points                    | +9.7 F1 points          | met           |
| P3        | Value head against Stockfish, Spearman ≥ 0.80                     | 0.666                   | **not met**   |
| P4        | Elo by condition monotonic: 1500 < 2000 < 2400, intervals apart   | neither, and measured   | **not met**   |
| P5        | DPO or GRPO ≥ +50 Elo over its base, interval included            | +44 (CI 23 to 66)       | **not met**   |
| P5        | Reward model ≥ 75 % on held-out pairs                             | 75.77 %                 | met           |
| P6        | One reproducible results table over every stage                   | `docs/benchmarks.md`    | met           |

Why the three missed ones were missed, and what came out of each: D-056 and D-077 for the value
correlation (the loss is worth four times what the capacity is, and neither closes 0.13 of
Spearman), D-100 to D-107 for the Elo by condition (what a fine-tune moves is *behaviour*, not
strength, and the module was rewritten around that), D-126 for the alignment Elo (the run that
publishes is not the one that scored highest, and the reason is in the decision).

## What each bar means exactly

**Legality without the mask.** The share of positions where the argmax of the raw distribution is a
legal move, with no temperature, no top-k and no legality mask. It is a property of the weights and
not of the sampler, which is why the demo's masked player cannot be read against it.
`rukh.eval.legality` measures it.

**Elo.** Games against Stockfish at `UCI_Elo` rungs, 0.1 s per move (below that the engine is not
calibrated at all, D-025), with a bootstrap interval. Every published row is measured at
`configs/eval/greedy.yaml`; temperature alone is worth about 180 Elo on the same weights (D-047),
so a number read at another sampling belongs in another table. An Elo *over a base* is read from a
head-to-head match in both directions (D-120), never by subtracting what each model did against a
third one.

**Blunder F1 and the margin.** The encoder's blunder head against a material-counting heuristic on
the same rows. A blunder is rare, so the threshold is chosen on a `tune` half and the F1 is scored
on a `score` half that shares no game with it: an F1 at a fixed 0.5 measures the operating point
rather than the representation (D-055).

**Value correlation.** Spearman, not Pearson, between the value head and Stockfish's centipawns:
the head is scored on the *order* of positions rather than on the scale, because the scale is
`tanh(cp / 400)` and the order is what a bar can be read on.

**Held-out pairs.** Accuracy of the reward model over preference pairs whose **games** it never saw
(`rukh.train.reward.split_examples`, and `rukh.train.dpo.split_pairs` for the same reason). Pairs of
one game are almost the same pair, so a split by pair measures memorisation.

## Where the numbers live

- The single results table: `docs/benchmarks.md`, rewritten by `rukh eval nightly`.
- Why a bar was missed and what was done instead: `docs/decisiones-de-ejecucion.md`.
- The published verdict per release: the "Acceptance bars" section of each model card.
