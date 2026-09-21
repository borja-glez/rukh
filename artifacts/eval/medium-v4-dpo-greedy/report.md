# Evaluation of `medium-v4-dpo-greedy`

- Suite: `full`
- Checkpoint: `checkpoints/medium-v4-dpo/dpo.pt`
- Weights SHA-256: `9105b36ecb58e31086012cf7b2eea3d5f734b3bcb0908168933f47e4b810e83f`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-21
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.8 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.8 % |
| Top-1 next move | 53.4 % |
| Top-3 next move | 80.5 % |
| Puzzles solved | 38.8 % |
| Estimated Elo | 1586 (95 % CI 1530-1651) |
| Mean centipawn loss | n/a |
| Opening diversity | 0.993 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 195 of 200 |
| Line entropy | 7.594 bits of 7.644 |
| Normalised | 0.993 |
| First-move entropy (no sampling) | 1.778 bits |

Most played lines:

| Line | Games |
|---|---:|
| `e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6` | 2 |
| `e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 e7e5 d4b5 d7d6 b1c3 a7a6` | 2 |
| `e2e4 d7d5 e4d5 d8d5 g1f3 c8g4 f1e2 b8c6 h2h3 g4f3 e2f3 d5e6` | 2 |
| `e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 d4c6 b7c6 f1d3 d7d5` | 2 |
| `e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6 f1d3 d8c7 e1g1 g8f6` | 2 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 998 | 99.8 % |
| sampled | 1000 | 998 | 99.8 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 49.9 % | 79.6 % |
| 2000-2200 | 328 | 54.9 % | 81.1 % |
| 2200-2400 | 97 | 62.9 % | 81.4 % |
| 2400-2600 | 28 | 67.9 % | 85.7 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1166 | 58.3 % |
| 1500-2000 | 2000 | 798 | 39.9 % |
| 2000+ | 2000 | 362 | 18.1 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 16 | 1 | 3 | 0.825 | 0 | 0 |
| skill-0 | 1381 | 20 | 16 | 3 | 1 | 0.875 | 0 | 0 |
| skill-1 | 1467 | 20 | 9 | 2 | 9 | 0.500 | 0 | 0 |
| uci-1500 | 1500 | 20 | 7 | 3 | 10 | 0.425 | 0 | 0 |
| skill-2 | 1589 | 20 | 10 | 1 | 9 | 0.525 | 0 | 0 |
| skill-3 | 1678 | 20 | 6 | 3 | 11 | 0.375 | 0 | 0 |
| uci-1800 | 1800 | 20 | 6 | 5 | 9 | 0.425 | 1 | 1 |
| uci-2000 | 2000 | 20 | 2 | 0 | 18 | 0.100 | 0 | 0 |

1 of 160 games hit the context limit; 1 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 1 of 160 games hit the context limit and were adjudicated (1 of them) instead of being scored as draws
