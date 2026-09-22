# Evaluation of `medium-elo@2100`

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
| Top-1 next move | 53.4 % |
| Top-3 next move | 81.9 % |
| Puzzles solved | 38.1 % |
| Estimated Elo | 1606 (95 % CI 1543-1670) |
| Mean centipawn loss | n/a |
| Opening diversity | 0.997 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 198 of 200 |
| Line entropy | 7.624 bits of 7.644 |
| Normalised | 0.997 |
| First-move entropy (no sampling) | 1.881 bits |

Most played lines:

| Line | Games |
|---|---:|
| `e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 d2d3 b7b5 a4b3 f8e7` | 2 |
| `e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6` | 2 |
| `d2d4 g8f6 g1f3 d7d5 c2c4 d5c4 b1c3 a7a6 a2a4 b8c6 e2e3 c6a5` | 1 |
| `e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 f1e2 d5f4 e1g1 f4e2 d1e2 c7c6` | 1 |
| `d2d4 g8f6 g1f3 g7g6 c2c4 f8g7 e2e3 e8g8 b1c3 d7d6 f1e2 b8d7` | 1 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `docs/acceptance.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 997 | 99.7 % |
| sampled | 1000 | 997 | 99.7 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 2000-2200 | 1000 | 53.4 % | 81.9 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1171 | 58.6 % |
| 1500-2000 | 2000 | 770 | 38.5 % |
| 2000+ | 2000 | 344 | 17.2 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 15 | 3 | 2 | 0.825 | 1 | 1 |
| skill-0 | 1381 | 20 | 14 | 1 | 5 | 0.725 | 0 | 0 |
| skill-1 | 1467 | 20 | 9 | 3 | 8 | 0.525 | 2 | 2 |
| uci-1500 | 1500 | 20 | 15 | 1 | 4 | 0.775 | 0 | 0 |
| skill-2 | 1589 | 20 | 10 | 0 | 10 | 0.500 | 0 | 0 |
| skill-3 | 1678 | 20 | 5 | 1 | 14 | 0.275 | 2 | 2 |
| uci-1800 | 1800 | 20 | 6 | 1 | 13 | 0.325 | 0 | 0 |
| uci-2000 | 2000 | 20 | 5 | 1 | 14 | 0.275 | 0 | 0 |

5 of 160 games hit the context limit; 5 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 5 of 160 games hit the context limit and were adjudicated (5 of them) instead of being scored as draws
