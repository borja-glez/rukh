# Evaluation of `medium-elo@1200`

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
| Top-1 next move | 54.9 % |
| Top-3 next move | 82.5 % |
| Puzzles solved | 37.0 % |
| Estimated Elo | 1425 (95 % CI 1361-1479) |
| Mean centipawn loss | n/a |
| Opening diversity | -0.000 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 1. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 1 of 200 |
| Line entropy | -0.000 bits of 7.644 |
| Normalised | -0.000 |
| First-move entropy (no sampling) | 1.596 bits |

Most played lines:

| Line | Games |
|---|---:|
| `e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 f3g5 d7d5 e4d5 f6d5 g5f7 e8f7` | 200 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 997 | 99.7 % |
| sampled | 1000 | 997 | 99.7 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 52.3 % | 80.7 % |
| 2000-2200 | 328 | 57.6 % | 84.1 % |
| 2200-2400 | 97 | 58.8 % | 85.6 % |
| 2400-2600 | 28 | 57.1 % | 85.7 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1239 | 62.0 % |
| 1500-2000 | 2000 | 693 | 34.6 % |
| 2000+ | 2000 | 289 | 14.4 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 13 | 3 | 4 | 0.725 | 1 | 1 |
| skill-0 | 1381 | 20 | 12 | 1 | 7 | 0.625 | 3 | 3 |
| skill-1 | 1467 | 20 | 10 | 2 | 8 | 0.550 | 0 | 0 |
| uci-1500 | 1500 | 20 | 8 | 0 | 12 | 0.400 | 1 | 1 |
| skill-2 | 1589 | 20 | 4 | 1 | 15 | 0.225 | 0 | 0 |
| skill-3 | 1678 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |
| uci-1800 | 1800 | 20 | 1 | 1 | 18 | 0.075 | 0 | 0 |
| uci-2000 | 2000 | 20 | 1 | 0 | 19 | 0.050 | 0 | 0 |

5 of 160 games hit the context limit; 5 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 5 of 160 games hit the context limit and were adjudicated (5 of them) instead of being scored as draws
