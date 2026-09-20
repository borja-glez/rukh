# Elo by condition of `medium-v4`

- Checkpoint: `checkpoints/medium-v4-20260919-174623/best.pt`
- Parameters: 115,120,128
- Date: 2026-09-20

Every row is the same suite with one number changed: the `<wXXXX> <bXXXX>` header the model is prompted with. Same opponents, same positions, same puzzles, same seed, same temperature.

| Asked for | Elo | 95 % CI | legal argmax | top-1 | puzzles | first-move entropy |
|---|---:|---|---:|---:|---:|---:|
| `<w1200>` | 1419 | 1358-1479 | 99.60 % | 49.80 % | 36.42 % | 2.856 bits |
| `<w2100>` | 1580 | 1512-1649 | 99.80 % | 53.80 % | 38.27 % | 1.903 bits |

## Verdict

- Point estimates: monotonic.
- Confidence intervals: separated.
- Span between the weakest and the strongest condition: 161 Elo.

The two are not the same claim. Point estimates rise by chance often enough that a monotonic row of numbers is not evidence on its own; the acceptance bar of `GOAL.md` is read on the intervals.
