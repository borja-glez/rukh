# Evaluation of `small`

- Suite: `quick`
- Checkpoint: `checkpoints/small-20260919-062911/best.pt`
- Weights SHA-256: `7130d64ac1c409332dfd9b753281b38a7b0b434103eb1908533942e30c243f4e`
- Parameters: 38,971,392
- Device: `cuda`
- Date: 2026-09-19
- MLflow run: be0f6cee4a6e4eaca9081b623bc73412

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.4 % |
| Legality without the mask, sampled (T=0.6, top-k 20) | 99.0 % |
| Top-1 next move | 51.1 % |
| Top-3 next move | 79.4 % |
| Puzzles solved | n/a |
| Estimated Elo | 785 (95 % CI 680-896) |
| Mean centipawn loss | n/a |
| Opening diversity | n/a |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `docs/acceptance.md`. **sampled** draws the token exactly as the demo does(T=0.6, top-k 20), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 994 | 99.4 % |
| sampled | 1000 | 990 | 99.0 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 49.9 % | 78.5 % |
| 2000-2200 | 328 | 52.7 % | 79.9 % |
| 2200-2400 | 97 | 52.6 % | 82.5 % |
| 2400-2600 | 28 | 50.0 % | 78.6 % |
| 2600+ | 4 | 50.0 % | 100.0 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| skill-0 | 800 | 20 | 6 | 0 | 14 | 0.300 | 1 | 1 |
| skill-1 | 950 | 20 | 2 | 2 | 16 | 0.150 | 0 | 0 |
| skill-2 | 1100 | 20 | 0 | 1 | 19 | 0.025 | 1 | 1 |
| skill-3 | 1250 | 20 | 2 | 0 | 18 | 0.100 | 1 | 1 |
| uci-1320 | 1320 | 20 | 6 | 0 | 14 | 0.300 | 1 | 1 |
| uci-1500 | 1500 | 20 | 2 | 0 | 18 | 0.100 | 1 | 1 |
| uci-1800 | 1800 | 20 | 1 | 0 | 19 | 0.050 | 0 | 0 |
| uci-2000 | 2000 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |

5 of 160 games hit the context limit; 5 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 0.6, top-k 20) and is always the lower of the two.
- puzzles not found at E:\work\ai\chess-lm\rukh\data\puzzles\puzzles.parquet: puzzle suite skipped
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 5 of 160 games hit the context limit and were adjudicated (5 of them) instead of being scored as draws
