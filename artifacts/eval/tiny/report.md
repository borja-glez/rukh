# Evaluation of `tiny`

- Suite: `quick`
- Checkpoint: `checkpoints/tiny-20260919-061533/best.pt`
- Weights SHA-256: `4591cf8cca0b91b38fc3c4969c7bbf18aa5d3b4b79b36558727b981c24df281a`
- Parameters: 5,309,952
- Device: `cuda`
- Date: 2026-09-19
- MLflow run: b1836b7e8934483693df84ff2f090a3d

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 94.5 % |
| Legality without the mask, sampled (T=0.6, top-k 20) | 93.8 % |
| Top-1 next move | 40.3 % |
| Top-3 next move | 67.1 % |
| Puzzles solved | n/a |
| Estimated Elo | 64 (95 % CI -200-292) |
| Mean centipawn loss | n/a |
| Opening diversity | n/a |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `docs/acceptance.md`. **sampled** draws the token exactly as the demo does(T=0.6, top-k 20), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 945 | 94.5 % |
| sampled | 1000 | 938 | 93.8 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 38.9 % | 66.1 % |
| 2000-2200 | 328 | 45.4 % | 71.0 % |
| 2200-2400 | 97 | 33.0 % | 59.8 % |
| 2400-2600 | 28 | 32.1 % | 64.3 % |
| 2600+ | 4 | 50.0 % | 75.0 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| skill-0 | 800 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |
| skill-1 | 950 | 20 | 0 | 1 | 19 | 0.025 | 0 | 0 |
| skill-2 | 1100 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |
| skill-3 | 1250 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |
| uci-1320 | 1320 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |
| uci-1500 | 1500 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |
| uci-1800 | 1800 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |
| uci-2000 | 2000 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |

0 of 160 games hit the context limit; 0 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 0.6, top-k 20) and is always the lower of the two.
- puzzles not found at E:\work\ai\chess-lm\rukh\data\puzzles\puzzles.parquet: puzzle suite skipped
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
