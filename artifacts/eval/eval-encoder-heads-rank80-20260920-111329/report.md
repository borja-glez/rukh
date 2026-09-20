# Evaluation of `eval-encoder-heads-rank80-20260920-111329`

- Suite: `encoder`
- Checkpoint: `checkpoints/encoder-heads-rank80-20260920-111329/step-4000.pt`
- Weights SHA-256: `110f27515d9168b4be514c57c623636234b6dc1e569397775961762952b1bab0`
- Parameters: 15,054,725
- Device: `cuda`
- Date: 2026-09-20
- MLflow run: 42e7283785ca4bb5af0eb39ca2dad0f8

## Headline

| Metric | Value |
|---|---|
| Blunder F1, encoder (tuned, `p >= 0.0732`) | 16.6 % |
| Blunder F1, encoder (fixed, `p >= 0.5`) | 0.0 % |
| Blunder F1, material baseline | 8.9 % |
| Margin over the baseline | +7.8 points |
| Margin >= 5 points | yes |
| Blunder ROC AUC | 0.735 |
| Blunder average precision | 0.104 |
| Blunder base rate | 3.7 % |
| Value vs `tanh(cp / 400)`, Spearman (headline) | 0.660 |
| Value vs `tanh(cp / 400)`, Pearson | 0.550 |
| Value correlation >= 0.80 | no |
| Result accuracy | 49.8 % |
| Positions | 10,000 (7,453 with a blunder label, 3,660 of them scored) |

## Blunder detection

A blunder is rare (3.7 % of the labelled rows), so **accuracy is meaningless** here: a model that always answers "no blunder" scores 96.3 % without knowing anything about chess. F1 at an arbitrary threshold is nearly as bad, because an uncalibrated sigmoid can rank the positions well and still put every probability below 0.5: that number measures the operating point, not the representation. This is why the threshold is chosen on a `tune` half and the F1 is reported on a `score` half, and why ROC AUC and average precision, which no threshold can flatter, are reported next to it.

The 7,453 labelled rows are cut in two by `game_id`, never by position, the policy the held-out split itself uses, because two positions of the same game are one move apart. The `tune` half (3,793 rows, 2,455 games) chose the threshold `p >= 0.0732` by maximising F1 there; the `score` half (3,660 rows, 2,368 games) is where every number below is measured, and no `game_id` is in both halves.

The material baseline is a hard yes/no rule: it has no threshold, so it was given no half to tune on and nothing was swept on its side. The margin below compares the model at its best operating point against the rule at its only one, and that is a courtesy the model receives, not one the baseline does.

| Detector | Operating point | Items | Blunders | Flagged | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| encoder | tuned, `p >= 0.0732` (chosen on `tune`) | 3660 | 144 | 337 | 11.9 % | 27.8 % | 16.6 % |
| encoder | fixed, `p >= 0.5` | 3660 | 144 | 0 | 0.0 % | 0.0 % | 0.0 % |
| heuristic | rule, no threshold to tune | 3660 | 144 | 2359 | 4.7 % | 77.1 % | 8.9 % |

The same threshold reaches an F1 of 16.2 % on the `tune` half it was chosen on. The distance between that and the `score` half above is what picking an operating point costs, and it is the reason the two halves are not the same rows.

### Without an operating point

ROC AUC is the probability that a random blunder is ranked above a random quiet move (0.5 is a coin flip); average precision is the area under the precision-recall curve, and a random ranking scores the base rate, which is printed next to it. Neither depends on a threshold, so neither can be flattered by choosing one. The span of the probabilities is here because it is what makes a fixed threshold reasonable or absurd.

| Detector | Items | Blunders | Base rate | ROC AUC | Average precision | Probability span |
|---|---:|---:|---:|---:|---:|---|
| encoder | 3660 | 144 | 3.9 % | 0.735 | 0.104 | 0.0021 to 0.2765 (mean 0.0338) |

## Value against Stockfish

Both predictors are correlated against `tanh(cp / 400)`, the bounded score the value head is trained on, never against raw `cp`: a forced mate is worth ±9 99x there and would decide Pearson for the whole set on its own. Spearman asks only whether the ranking is right and is the number the goal is checked against; Pearson also asks whether the scale is.

| Predictor | Items | Target | Spearman | Pearson |
|---|---:|---|---:|---:|
| encoder | 10000 | `tanh(cp / 400)` | 0.660 | 0.550 |
| heuristic | 10000 | `tanh(cp / 400)` | 0.316 | 0.545 |

## Notes

- the blunder F1 of the encoder and of the material baseline are measured on the same rows of the held-out 'val' split, which is drawn by game_id, never by position, and those rows are cut in two by game_id again: the 'tune' half chooses the threshold and the 'score' half is what gets reported
- the checkpoint carries no label-count curve: run `rukh train heads --curve` and evaluate one of its checkpoints to fill that table
- the encoder is +7.8 F1 points from the baseline; GOAL.md asks for at least +5
- the value head is correlated against `tanh(cp / 400)`, the bounded score it is trained on, and not against raw `cp`, where a forced mate is worth ±9 99x and a few rows would decide Pearson for the whole set; the GOAL.md bar of 0.80 is read on Spearman
- the baseline counts material (1/3/3/5/9) and mobility and looks one ply ahead at captures: it cannot see a positional sacrifice, and calls Fischer's 17...Be6 (Byrne-Fischer, 1956) a nine-point blunder
- the two blunder detectors do not see the same thing: the baseline is given the predecessor position and the move that was played, while the encoder is given only the resulting position and has to infer that something was thrown away. That is the comparison GOAL.md asks for, but it is not a level playing field
- the blunder threshold 0.0732 was chosen on the 'tune' half (3,793 rows, 2,455 games) by maximising F1 there, and the reported numbers are measured on the 'score' half (3,660 rows, 2,368 games), which no threshold ever saw; the same rows at the fixed threshold 0.5 give an F1 of 0.0000 against 0.1663 tuned
- the material baseline is a hard yes/no rule: it has no threshold, so nothing was tuned on its side and it got no half to tune on. The margin therefore compares a model at its best operating point against a rule at its only one
- a blunder is rare (3.7 % of the labelled rows), so **accuracy is meaningless** here: a model that always answers "no blunder" scores 96.3 % without knowing anything about chess. F1 at an arbitrary threshold is nearly as bad, because an uncalibrated sigmoid can rank the positions well and still put every probability below 0.5: that number measures the operating point, not the representation. This is why the threshold is chosen on a `tune` half and the F1 is reported on a `score` half, and why ROC AUC and average precision, which no threshold can flatter, are reported next to it
