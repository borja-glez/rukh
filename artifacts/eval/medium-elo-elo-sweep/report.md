# Elo by condition of `medium-elo`

- Checkpoint: `checkpoints/medium-elo-20260920-143756/step-3800.pt`
- Parameters: 115,120,128
- Date: 2026-09-20

Every row is the same suite with one number changed: the `<wXXXX> <bXXXX>` header the model is prompted with. Same opponents, same positions, same puzzles, same seed, same temperature.

| Asked for | Elo | 95 % CI | legal argmax | top-1 | puzzles | first-move entropy |
|---|---:|---|---:|---:|---:|---:|
| `<w1200>` | 1425 | 1361-1479 | 100.00 % | 51.40 % | 37.02 % | 1.596 bits |
| `<w1500>` | 1549 | 1481-1609 | 100.00 % | 53.90 % | 37.67 % | 1.647 bits |
| `<w1800>` | 1498 | 1435-1552 | 99.80 % | 54.10 % | 38.20 % | 1.741 bits |
| `<w2000>` | 1538 | 1479-1597 | 99.70 % | 54.00 % | 37.98 % | 1.841 bits |
| `<w2100>` | 1606 | 1543-1670 | 99.70 % | 53.40 % | 38.08 % | 1.881 bits |
| `<w2400>` | 1644 | 1578-1707 | 99.70 % | 53.30 % | 38.08 % | 1.992 bits |

## Verdict

- Point estimates: **not** monotonic.
- Confidence intervals: **not** separated.
- Span between the weakest and the strongest condition: 219 Elo.

The two are not the same claim. Point estimates rise by chance often enough that a monotonic row of numbers is not evidence on its own; the acceptance bar of `docs/acceptance.md` is read on the intervals.
