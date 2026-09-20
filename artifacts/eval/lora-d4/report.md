# Evaluation of `lora-d4`

- Suite: `full`
- Checkpoint: `checkpoints/lora-d4-20260920-160902/step-1500.pt`
- Weights SHA-256: `e3876f4119b656fd585c5ffddf5c3878dbb8b161a767f694121cf31ce79da154`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-20
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.8 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.8 % |
| Top-1 next move | 55.1 % |
| Top-3 next move | 83.0 % |
| Puzzles solved | 37.3 % |
| Estimated Elo | n/a |
| Mean centipawn loss | n/a |
| Opening diversity | 1.000 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 200 of 200 |
| Line entropy | 7.644 bits of 7.644 |
| Normalised | 1.000 |
| First-move entropy (no sampling) | 0.016 bits |

Most played lines:

| Line | Games |
|---|---:|
| `d2d4 g8f6 g1f3 d7d5 e2e3 c8f5 f1d3 f5g4 c2c3 b8d7 b1d2 e7e5` | 1 |
| `d2d4 d7d5 c2c4 c7c6 b1c3 g8f6 c1g5 e7e6 e2e3 d8a5 g5f6 g7f6` | 1 |
| `d2d4 g8f6 g1f3 g7g6 e2e3 f8g7 b2b4 e8g8 c1b2 d7d6 c2c4 b8d7` | 1 |
| `d2d4 a7a6 f2f3 b7b5 e2e4 c8b7 d4d5 c7c5 c2c4 b5b4 b2b3 d7d6` | 1 |
| `d2d4 e7e6 c2c4 d7d5 b1c3 f8b4 g1f3 d5c4 e2e4 g8f6 c1g5 b8d7` | 1 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 998 | 99.8 % |
| sampled | 1000 | 998 | 99.8 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 51.9 % | 82.3 % |
| 2000-2200 | 328 | 57.3 % | 84.1 % |
| 2200-2400 | 97 | 61.9 % | 82.5 % |
| 2400-2600 | 28 | 64.3 % | 82.1 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1147 | 57.4 % |
| 1500-2000 | 2000 | 751 | 37.5 % |
| 2000+ | 2000 | 338 | 16.9 % |

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- Elo skipped: no games were played
