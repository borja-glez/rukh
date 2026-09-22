# Evaluation of `medium-elo@2000`

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
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.7 % |
| Top-1 next move | 54.0 % |
| Top-3 next move | 82.0 % |
| Puzzles solved | 38.0 % |
| Estimated Elo | 1538 (95 % CI 1479-1597) |
| Mean centipawn loss | n/a |
| Opening diversity | 1.000 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 200 of 200 |
| Line entropy | 7.644 bits of 7.644 |
| Normalised | 1.000 |
| First-move entropy (no sampling) | 1.841 bits |

Most played lines:

| Line | Games |
|---|---:|
| `d2d4 g8f6 g1f3 d7d5 g2g3 c8f5 f1g2 e7e6 e1g1 c7c5 c2c4 d5c4` | 1 |
| `e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 f1e2 d5f4 e1g1 f4e2 d1e2 c7c6` | 1 |
| `d2d4 g8f6 g1f3 g7g6 g2g3 f8g7 f1g2 e8g8 e1g1 d7d6 c2c4 b8d7` | 1 |
| `g1f3 a7a6 g2g3 b7b5 f1g2 c8b7 e1g1 c7c5 d2d3 e7e6 f3h4 b7g2` | 1 |
| `e2e4 e7e6 d2d4 d7d5 e4d5 e6d5 f1d3 g8f6 g1f3 f8d6 d1e2 c8e6` | 1 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `docs/acceptance.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 997 | 99.7 % |
| sampled | 1000 | 997 | 99.7 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 2000-2200 | 1000 | 54.0 % | 82.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1180 | 59.0 % |
| 1500-2000 | 2000 | 762 | 38.1 % |
| 2000+ | 2000 | 337 | 16.9 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 16 | 0 | 4 | 0.800 | 1 | 1 |
| skill-0 | 1381 | 20 | 13 | 3 | 4 | 0.725 | 0 | 0 |
| skill-1 | 1467 | 20 | 12 | 2 | 6 | 0.650 | 0 | 0 |
| uci-1500 | 1500 | 20 | 8 | 2 | 10 | 0.450 | 1 | 1 |
| skill-2 | 1589 | 20 | 7 | 1 | 12 | 0.375 | 0 | 0 |
| skill-3 | 1678 | 20 | 5 | 3 | 12 | 0.325 | 0 | 0 |
| uci-1800 | 1800 | 20 | 3 | 1 | 16 | 0.175 | 0 | 0 |
| uci-2000 | 2000 | 20 | 2 | 1 | 17 | 0.125 | 0 | 0 |

2 of 160 games hit the context limit; 2 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 2 of 160 games hit the context limit and were adjudicated (2 of them) instead of being scored as draws
