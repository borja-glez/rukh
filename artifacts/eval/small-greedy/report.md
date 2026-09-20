# Evaluation of `small-greedy`

- Suite: `full`
- Checkpoint: `checkpoints/small-20260919-062911/best.pt`
- Weights SHA-256: `7130d64ac1c409332dfd9b753281b38a7b0b434103eb1908533942e30c243f4e`
- Parameters: 38,971,392
- Device: `cuda`
- Date: 2026-09-20
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.4 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.4 % |
| Top-1 next move | 51.1 % |
| Top-3 next move | 79.4 % |
| Puzzles solved | 22.1 % |
| Estimated Elo | 1321 (95 % CI 1247-1383) |
| Mean centipawn loss | n/a |
| Opening diversity | n/a |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 994 | 99.4 % |
| sampled | 1000 | 994 | 99.4 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 49.9 % | 78.5 % |
| 2000-2200 | 328 | 52.7 % | 79.9 % |
| 2200-2400 | 97 | 52.6 % | 82.5 % |
| 2400-2600 | 28 | 50.0 % | 78.6 % |
| 2600+ | 4 | 50.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 694 | 34.7 % |
| 1500-2000 | 2000 | 423 | 21.1 % |
| 2000+ | 2000 | 207 | 10.3 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 9 | 0 | 11 | 0.450 | 1 | 1 |
| skill-0 | 1381 | 20 | 9 | 1 | 10 | 0.475 | 2 | 2 |
| skill-1 | 1467 | 20 | 2 | 5 | 13 | 0.225 | 1 | 1 |
| uci-1500 | 1500 | 20 | 6 | 1 | 13 | 0.325 | 0 | 0 |
| skill-2 | 1589 | 20 | 3 | 1 | 16 | 0.175 | 0 | 0 |
| skill-3 | 1678 | 20 | 1 | 2 | 17 | 0.100 | 2 | 2 |
| uci-1800 | 1800 | 20 | 1 | 1 | 18 | 0.075 | 0 | 0 |
| uci-2000 | 2000 | 20 | 0 | 1 | 19 | 0.025 | 0 | 0 |

6 of 160 games hit the context limit; 6 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 6 of 160 games hit the context limit and were adjudicated (6 of them) instead of being scored as draws
