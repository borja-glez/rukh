# Evaluation of `medium-v4-greedy`

- Suite: `full`
- Checkpoint: `E:/work/ai/chess-lm/rukh/checkpoints/medium-v4-20260919-174623/best.pt`
- Weights SHA-256: `98e05fcdfa2d1619ba33294e2d7f775c677b898e947b570ec2633de0e824b85e`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-21
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.8 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.8 % |
| Top-1 next move | 54.4 % |
| Top-3 next move | 82.2 % |
| Puzzles solved | 37.5 % |
| Estimated Elo | 1535 (95 % CI 1476-1599) |
| Mean centipawn loss | n/a |
| Opening diversity | 1.000 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 200 of 200 |
| Line entropy | 7.644 bits of 7.644 |
| Normalised | 1.000 |
| First-move entropy (no sampling) | 1.769 bits |

Most played lines:

| Line | Games |
|---|---:|
| `d2d4 g8f6 g1f3 d7d5 e2e3 c8f5 f1d3 f5g4 c2c3 b8d7 b1d2 e7e5` | 1 |
| `e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 f1e2 b8c6 e1g1 c8f5 d2d3 e7e5` | 1 |
| `d2d4 g8f6 g1f3 g7g6 g2g3 f8g7 f1g2 e8g8 e1g1 d7d6 c2c4 b8d7` | 1 |
| `g1f3 a7a6 d2d3 b7b5 c2c3 c8b7 b1d2 c7c5 g2g3 e7e6 f1g2 f8e7` | 1 |
| `e2e4 e7e6 d2d4 d7d5 e4d5 e6d5 f1d3 c7c6 g1f3 g8f6 d1e2 f8e7` | 1 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `docs/acceptance.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 998 | 99.8 % |
| sampled | 1000 | 998 | 99.8 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 51.6 % | 80.7 % |
| 2000-2200 | 328 | 56.4 % | 84.1 % |
| 2200-2400 | 97 | 60.8 % | 83.5 % |
| 2400-2600 | 28 | 60.7 % | 82.1 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1158 | 57.9 % |
| 1500-2000 | 2000 | 761 | 38.0 % |
| 2000+ | 2000 | 331 | 16.6 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 15 | 1 | 4 | 0.775 | 1 | 1 |
| skill-0 | 1381 | 20 | 13 | 1 | 6 | 0.675 | 2 | 2 |
| skill-1 | 1467 | 20 | 12 | 1 | 7 | 0.625 | 1 | 1 |
| uci-1500 | 1500 | 20 | 7 | 3 | 10 | 0.425 | 0 | 0 |
| skill-2 | 1589 | 20 | 6 | 2 | 12 | 0.350 | 0 | 0 |
| skill-3 | 1678 | 20 | 6 | 3 | 11 | 0.375 | 0 | 0 |
| uci-1800 | 1800 | 20 | 3 | 2 | 15 | 0.200 | 0 | 0 |
| uci-2000 | 2000 | 20 | 3 | 1 | 16 | 0.175 | 1 | 1 |

5 of 160 games hit the context limit; 5 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 5 of 160 games hit the context limit and were adjudicated (5 of them) instead of being scored as draws
