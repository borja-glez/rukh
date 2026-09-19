# Evaluation of `medium-v4-greedy`

- Suite: `full`
- Checkpoint: `checkpoints/medium-v4-20260919-174623/best.pt`
- Weights SHA-256: `98e05fcdfa2d1619ba33294e2d7f775c677b898e947b570ec2633de0e824b85e`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-19
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.8 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.8 % |
| Top-1 next move | 54.4 % |
| Top-3 next move | 82.2 % |
| Puzzles solved | 37.5 % |
| Estimated Elo | 1504 (95 % CI 1446-1558) |
| Mean centipawn loss | n/a |
| Opening diversity | n/a |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 998 | 99.8 % |
| sampled | 1000 | 998 | 99.8 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 51.6 % | 80.7 % |
| 2000-2200 | 328 | 56.4 % | 84.1 % |
| 2200-2400 | 97 | 60.8 % | 83.5 % |
| 2400-2600 | 28 | 60.7 % | 82.1 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1158 | 57.9 % |
| 1500-2000 | 2000 | 761 | 38.0 % |
| 2000+ | 2000 | 331 | 16.6 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 15 | 2 | 3 | 0.800 | 0 | 0 |
| skill-0 | 1381 | 20 | 16 | 1 | 3 | 0.825 | 1 | 1 |
| skill-1 | 1467 | 20 | 9 | 2 | 9 | 0.500 | 0 | 0 |
| uci-1500 | 1500 | 20 | 3 | 4 | 13 | 0.250 | 1 | 1 |
| skill-2 | 1589 | 20 | 6 | 3 | 11 | 0.375 | 0 | 0 |
| skill-3 | 1678 | 20 | 5 | 2 | 13 | 0.300 | 0 | 0 |
| uci-1800 | 1800 | 20 | 2 | 2 | 16 | 0.150 | 0 | 0 |
| uci-2000 | 2000 | 20 | 2 | 1 | 17 | 0.125 | 0 | 0 |

2 of 160 games hit the context limit; 2 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 2 of 160 games hit the context limit and were adjudicated (2 of them) instead of being scored as draws
