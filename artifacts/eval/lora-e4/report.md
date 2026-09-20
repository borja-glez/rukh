# Evaluation of `lora-e4`

- Suite: `full`
- Checkpoint: `checkpoints/lora-e4-20260920-160133/step-1500.pt`
- Weights SHA-256: `da7025c4de105d00b8c6fe70f3f64b02781bcde993be138efee26cca8e74eaa6`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-20
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.8 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.8 % |
| Top-1 next move | 54.7 % |
| Top-3 next move | 83.3 % |
| Puzzles solved | 37.8 % |
| Estimated Elo | n/a |
| Mean centipawn loss | n/a |
| Opening diversity | 0.999 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 199 of 200 |
| Line entropy | 7.634 bits of 7.644 |
| Normalised | 0.999 |
| First-move entropy (no sampling) | 0.021 bits |

Most played lines:

| Line | Games |
|---|---:|
| `e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 c5d4 c3d4 f8b4` | 2 |
| `e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f8c5 c2c3 a7a6 b5a4 b7b5` | 1 |
| `e2e4 d7d5 e4d5 g8f6 f1c4 f6d5 g1f3 c8g4 e1g1 e7e6 d2d3 c7c6` | 1 |
| `e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1c4 e7e6` | 1 |
| `e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5` | 1 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 998 | 99.8 % |
| sampled | 1000 | 998 | 99.8 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 51.9 % | 82.1 % |
| 2000-2200 | 328 | 56.4 % | 84.5 % |
| 2200-2400 | 97 | 60.8 % | 84.5 % |
| 2400-2600 | 28 | 64.3 % | 85.7 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1162 | 58.1 % |
| 1500-2000 | 2000 | 764 | 38.2 % |
| 2000+ | 2000 | 343 | 17.2 % |

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- Elo skipped: no games were played
