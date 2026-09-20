# Evaluation of `small-v3-greedy`

- Suite: `full`
- Checkpoint: `checkpoints/small-v3-20260919-160710/best.pt`
- Weights SHA-256: `3104796b3e7fa43383d8c43924f06d32db7b9e6cd2352d2ee6c8b62cfaf36247`
- Parameters: 38,971,392
- Device: `cuda`
- Date: 2026-09-20
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.1 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.1 % |
| Top-1 next move | 52.4 % |
| Top-3 next move | 80.5 % |
| Puzzles solved | 26.7 % |
| Estimated Elo | 1365 (95 % CI 1293-1423) |
| Mean centipawn loss | n/a |
| Opening diversity | n/a |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 991 | 99.1 % |
| sampled | 1000 | 991 | 99.1 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 52.5 % | 79.7 % |
| 2000-2200 | 328 | 50.6 % | 82.0 % |
| 2200-2400 | 97 | 59.8 % | 80.4 % |
| 2400-2600 | 28 | 42.9 % | 75.0 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 820 | 41.0 % |
| 1500-2000 | 2000 | 540 | 27.0 % |
| 2000+ | 2000 | 244 | 12.2 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 8 | 1 | 11 | 0.425 | 1 | 1 |
| skill-0 | 1381 | 20 | 13 | 0 | 7 | 0.650 | 0 | 0 |
| skill-1 | 1467 | 20 | 5 | 2 | 13 | 0.300 | 1 | 1 |
| uci-1500 | 1500 | 20 | 4 | 0 | 16 | 0.200 | 0 | 0 |
| skill-2 | 1589 | 20 | 5 | 2 | 13 | 0.300 | 1 | 1 |
| skill-3 | 1678 | 20 | 4 | 1 | 15 | 0.225 | 0 | 0 |
| uci-1800 | 1800 | 20 | 0 | 1 | 19 | 0.025 | 0 | 0 |
| uci-2000 | 2000 | 20 | 1 | 0 | 19 | 0.050 | 0 | 0 |

3 of 160 games hit the context limit; 3 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 3 of 160 games hit the context limit and were adjudicated (3 of them) instead of being scored as draws
