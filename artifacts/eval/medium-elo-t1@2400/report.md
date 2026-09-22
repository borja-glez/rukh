# Evaluation of `medium-elo-t1@2400`

- Suite: `full`
- Checkpoint: `checkpoints/medium-elo-20260920-143756/step-3800.pt`
- Weights SHA-256: `bcb67028547ca0037634d3fdd97862f9251abe2fe74da0cdcc7586668b755cf6`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-20
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.7 % |
| Legality without the mask, sampled (T=1, top-k 20) | 99.0 % |
| Top-1 next move | 53.3 % |
| Top-3 next move | 81.4 % |
| Puzzles solved | 38.1 % |
| Estimated Elo | 1272 (95 % CI 1193-1340) |
| Mean centipawn loss | n/a |
| Opening diversity | 0.997 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 198 of 200 |
| Line entropy | 7.620 bits of 7.644 |
| Normalised | 0.997 |
| First-move entropy (no sampling) | 1.992 bits |

Most played lines:

| Line | Games |
|---|---:|
| `e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6` | 3 |
| `d2d4 g8f6 g1f3 d7d5 c2c4 d5c4 b1c3 a7a6 a2a4 b8c6 e2e3 c6a5` | 1 |
| `e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 g7g6 c2c4 d5b6 c4c5 b6d5` | 1 |
| `d2d4 g8f6 g1f3 g7g6 c2c4 f8g7 b1c3 e8g8 e2e4 d7d6 h2h3 e7e5` | 1 |
| `g1f3 a7a6 g2g3 b7b5 f1g2 c8b7 e1g1 c7c5 d2d3 e7e6 f3h4 b7g2` | 1 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `docs/acceptance.md`. **sampled** draws the token exactly as the demo does(T=1, top-k 20), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 997 | 99.7 % |
| sampled | 1000 | 990 | 99.0 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 2400-2600 | 1000 | 53.3 % | 81.4 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1156 | 57.8 % |
| 1500-2000 | 2000 | 774 | 38.7 % |
| 2000+ | 2000 | 355 | 17.8 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 5 | 3 | 12 | 0.325 | 0 | 0 |
| skill-0 | 1381 | 20 | 8 | 2 | 10 | 0.450 | 1 | 1 |
| skill-1 | 1467 | 20 | 4 | 1 | 15 | 0.225 | 1 | 1 |
| uci-1500 | 1500 | 20 | 7 | 0 | 13 | 0.350 | 0 | 0 |
| skill-2 | 1589 | 20 | 1 | 0 | 19 | 0.050 | 1 | 1 |
| skill-3 | 1678 | 20 | 1 | 0 | 19 | 0.050 | 0 | 0 |
| uci-1800 | 1800 | 20 | 1 | 1 | 18 | 0.075 | 0 | 0 |
| uci-2000 | 2000 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |

3 of 160 games hit the context limit; 3 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 1, top-k 20) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 3 of 160 games hit the context limit and were adjudicated (3 of them) instead of being scored as draws
