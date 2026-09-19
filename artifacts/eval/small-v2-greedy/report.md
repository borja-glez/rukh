# Evaluation of `small-v2-greedy`

- Suite: `full`
- Checkpoint: `checkpoints/small-v2-20260919-150832/best.pt`
- Weights SHA-256: `ed324252068892d2f20498c83e45496ac93e1457c79fd79099ff16d0a83db3cd`
- Parameters: 38,971,392
- Device: `cuda`
- Date: 2026-09-19
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.3 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.3 % |
| Top-1 next move | 51.8 % |
| Top-3 next move | 80.6 % |
| Puzzles solved | 26.9 % |
| Estimated Elo | 1070 (95 % CI 975-1167) |
| Mean centipawn loss | n/a |
| Opening diversity | n/a |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 993 | 99.3 % |
| sampled | 1000 | 993 | 99.3 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 51.4 % | 79.4 % |
| 2000-2200 | 328 | 52.1 % | 83.2 % |
| 2200-2400 | 97 | 54.6 % | 79.4 % |
| 2400-2600 | 28 | 42.9 % | 75.0 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 868 | 43.4 % |
| 1500-2000 | 2000 | 520 | 26.0 % |
| 2000+ | 2000 | 228 | 11.4 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| skill-0 | 800 | 20 | 9 | 1 | 10 | 0.475 | 0 | 0 |
| skill-1 | 950 | 20 | 8 | 2 | 10 | 0.450 | 1 | 1 |
| skill-2 | 1100 | 20 | 4 | 3 | 13 | 0.275 | 2 | 2 |
| skill-3 | 1250 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |
| uci-1320 | 1320 | 20 | 14 | 1 | 5 | 0.725 | 0 | 0 |
| uci-1500 | 1500 | 20 | 6 | 3 | 11 | 0.375 | 0 | 0 |
| uci-1800 | 1800 | 20 | 1 | 3 | 16 | 0.125 | 0 | 0 |
| uci-2000 | 2000 | 20 | 0 | 3 | 17 | 0.075 | 0 | 0 |

3 of 160 games hit the context limit; 3 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 3 of 160 games hit the context limit and were adjudicated (3 of them) instead of being scored as draws
