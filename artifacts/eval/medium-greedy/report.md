# Evaluation of `medium-greedy`

- Suite: `full`
- Checkpoint: `checkpoints/medium-20260919-073839/best.pt`
- Weights SHA-256: `f7654f4e0842ba5986f3cefc1d19ebcc54da8c1061e5130333163e772401a27f`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-19
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.4 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.4 % |
| Top-1 next move | 52.9 % |
| Top-3 next move | 80.9 % |
| Puzzles solved | 23.9 % |
| Estimated Elo | 1091 (95 % CI 990-1194) |
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
| 1800-2000 | 543 | 52.5 % | 80.8 % |
| 2000-2200 | 328 | 54.6 % | 81.4 % |
| 2200-2400 | 97 | 48.5 % | 81.4 % |
| 2400-2600 | 28 | 53.6 % | 71.4 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 752 | 37.6 % |
| 1500-2000 | 2000 | 480 | 24.0 % |
| 2000+ | 2000 | 204 | 10.2 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| skill-0 | 800 | 20 | 10 | 2 | 8 | 0.550 | 0 | 0 |
| skill-1 | 950 | 20 | 7 | 1 | 12 | 0.375 | 0 | 0 |
| skill-2 | 1100 | 20 | 5 | 1 | 14 | 0.275 | 0 | 0 |
| skill-3 | 1250 | 20 | 1 | 0 | 19 | 0.050 | 2 | 2 |
| uci-1320 | 1320 | 20 | 12 | 2 | 6 | 0.650 | 0 | 0 |
| uci-1500 | 1500 | 20 | 7 | 1 | 12 | 0.375 | 0 | 0 |
| uci-1800 | 1800 | 20 | 3 | 5 | 12 | 0.275 | 1 | 1 |
| uci-2000 | 2000 | 20 | 1 | 1 | 18 | 0.075 | 1 | 1 |

4 of 160 games hit the context limit; 4 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 4 of 160 games hit the context limit and were adjudicated (4 of them) instead of being scored as draws
