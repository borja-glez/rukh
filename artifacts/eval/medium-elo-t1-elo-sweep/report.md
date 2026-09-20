# Elo by condition of `medium-elo-t1`

- Checkpoint: `checkpoints/medium-elo-20260920-143756/step-3800.pt`
- Parameters: 115,120,128
- Date: 2026-09-20

Every row is the same suite with one number changed: the `<wXXXX> <bXXXX>` header the model is prompted with. Same opponents, same positions, same puzzles, same seed, same temperature.

| Asked for | Elo | 95 % CI | legal argmax | top-1 | puzzles | first-move entropy |
|---|---:|---|---:|---:|---:|---:|
| `<w1200>` | 1002 | 866-1109 | 100.00 % | 51.40 % | 37.02 % | 1.596 bits |
| `<w2400>` | 1272 | 1193-1340 | 99.70 % | 53.30 % | 38.08 % | 1.992 bits |

## Verdict

- Point estimates: monotonic.
- Confidence intervals: separated.
- Span between the weakest and the strongest condition: 270 Elo.

The two are not the same claim. Point estimates rise by chance often enough that a monotonic row of numbers is not evidence on its own; the acceptance bar of `GOAL.md` is read on the intervals.
