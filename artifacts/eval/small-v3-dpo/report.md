# Evaluation of `small-v3-dpo`

- Suite: `full`
- Checkpoint: `checkpoints/small-v3-dpo-anchored/dpo.pt`
- Weights SHA-256: `b65a39a70c53340eac5b62e9d13b7c38d0350861e834273e940bc7100fc138c3`
- Parameters: 38,971,392
- Device: `cuda`
- Date: 2026-09-19
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 98.9 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 98.9 % |
| Top-1 next move | 53.2 % |
| Top-3 next move | 79.3 % |
| Puzzles solved | 27.9 % |
| Estimated Elo | 1143 (95 % CI 1047-1239) |
| Mean centipawn loss | n/a |
| Opening diversity | n/a |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 989 | 98.9 % |
| sampled | 1000 | 989 | 98.9 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 53.0 % | 78.1 % |
| 2000-2200 | 328 | 50.9 % | 80.5 % |
| 2200-2400 | 97 | 61.9 % | 81.4 % |
| 2400-2600 | 28 | 46.4 % | 78.6 % |
| 2600+ | 4 | 100.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 846 | 42.3 % |
| 1500-2000 | 2000 | 568 | 28.4 % |
| 2000+ | 2000 | 258 | 12.9 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| skill-0 | 800 | 20 | 14 | 2 | 4 | 0.750 | 2 | 2 |
| skill-1 | 950 | 20 | 5 | 2 | 13 | 0.300 | 0 | 0 |
| skill-2 | 1100 | 20 | 4 | 1 | 15 | 0.225 | 2 | 2 |
| skill-3 | 1250 | 20 | 0 | 2 | 18 | 0.050 | 0 | 0 |
| uci-1320 | 1320 | 20 | 15 | 1 | 4 | 0.775 | 1 | 1 |
| uci-1500 | 1500 | 20 | 9 | 3 | 8 | 0.525 | 0 | 0 |
| uci-1800 | 1800 | 20 | 2 | 1 | 17 | 0.125 | 0 | 0 |
| uci-2000 | 2000 | 20 | 4 | 0 | 16 | 0.200 | 1 | 1 |

6 of 160 games hit the context limit; 6 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 6 of 160 games hit the context limit and were adjudicated (6 of them) instead of being scored as draws
