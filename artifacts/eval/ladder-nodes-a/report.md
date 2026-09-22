# Evaluation of `ladder-nodes-a`

- Suite: `full`
- Checkpoint: `checkpoints/medium-v4-20260919-174623/best.pt`
- Weights SHA-256: `98e05fcdfa2d1619ba33294e2d7f775c677b898e947b570ec2633de0e824b85e`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-21
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | n/a |
| Legality without the mask, sampled | n/a |
| Top-1 next move | n/a |
| Top-3 next move | n/a |
| Puzzles solved | 0.0 % |
| Estimated Elo | 1541 (95 % CI 1485-1596) |
| Mean centipawn loss | n/a |
| Opening diversity | n/a |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 14 | 3 | 3 | 0.775 | 0 | 0 |
| skill-0 | 1381 | 20 | 13 | 2 | 5 | 0.700 | 2 | 2 |
| skill-1 | 1467 | 20 | 15 | 1 | 4 | 0.775 | 0 | 0 |
| uci-1500 | 1500 | 20 | 5 | 3 | 12 | 0.325 | 1 | 1 |
| skill-2 | 1589 | 20 | 8 | 2 | 10 | 0.450 | 1 | 1 |
| skill-3 | 1678 | 20 | 5 | 3 | 12 | 0.325 | 0 | 0 |
| uci-1800 | 1800 | 20 | 2 | 3 | 15 | 0.175 | 0 | 0 |
| uci-2000 | 2000 | 20 | 2 | 1 | 17 | 0.125 | 0 | 0 |

4 of 160 games hit the context limit; 4 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the puzzle parquet has no ``prefix_uci``, so every puzzle was prompted with its own solution line after ``<bos>``: a token sequence that is no game and does not start from the initial position. The rate is a floor, not a measurement, and is not comparable with a run scored from the real game prefix (rebuild the parquet with ``rukh data puzzles``)
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 4 of 160 games hit the context limit and were adjudicated (4 of them) instead of being scored as draws
