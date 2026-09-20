# Evaluation of `tiny-greedy`

- Suite: `full`
- Checkpoint: `checkpoints/tiny-20260919-061533/best.pt`
- Weights SHA-256: `4591cf8cca0b91b38fc3c4969c7bbf18aa5d3b4b79b36558727b981c24df281a`
- Parameters: 5,309,952
- Device: `cuda`
- Date: 2026-09-20
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 94.5 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 94.5 % |
| Top-1 next move | 40.3 % |
| Top-3 next move | 67.1 % |
| Puzzles solved | 8.9 % |
| Estimated Elo | 921 (95 % CI 713-1040) |
| Mean centipawn loss | n/a |
| Opening diversity | 1.000 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 200 of 200 |
| Line entropy | 7.644 bits of 7.644 |
| Normalised | 1.000 |
| First-move entropy (no sampling) | 1.721 bits |

Most played lines:

| Line | Games |
|---|---:|
| `d2d4 g8f6 g1f3 d7d5 c2c4 d5c4 b1c3 c8g4 e2e3 c7c6 f1c4 e7e6` | 1 |
| `e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 f1e2 b8c6 e1g1 c8f5 d2d3 e7e5` | 1 |
| `d2d4 g8f6 g1f3 g7g6 e2e3 f8g7 f1e2 e8g8 e1g1 d7d6 b2b3 b8d7` | 1 |
| `g1f3 a7a6 d2d3 b7b5 e2e4 c8b7 g2g3 c7c5 f1g2 e7e6 e1g1 f8e7` | 1 |
| `e2e4 e7e6 d2d4 d7d5 e4d5 e6d5 g1f3 c8g4 f1e2 g8f6 c1g5 f8d6` | 1 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 945 | 94.5 % |
| sampled | 1000 | 945 | 94.5 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 38.9 % | 66.1 % |
| 2000-2200 | 328 | 45.4 % | 71.0 % |
| 2200-2400 | 97 | 33.0 % | 59.8 % |
| 2400-2600 | 28 | 32.1 % | 64.3 % |
| 2600+ | 4 | 50.0 % | 75.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 283 | 14.1 % |
| 1500-2000 | 2000 | 173 | 8.6 % |
| 2000+ | 2000 | 77 | 3.9 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 2 | 1 | 17 | 0.125 | 1 | 1 |
| skill-0 | 1381 | 20 | 0 | 1 | 19 | 0.025 | 0 | 0 |
| skill-1 | 1467 | 20 | 1 | 0 | 19 | 0.050 | 0 | 0 |
| uci-1500 | 1500 | 20 | 1 | 0 | 19 | 0.050 | 0 | 0 |
| skill-2 | 1589 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |
| skill-3 | 1678 | 20 | 0 | 0 | 20 | 0.000 | 1 | 1 |
| uci-1800 | 1800 | 20 | 0 | 0 | 20 | 0.000 | 0 | 0 |
| uci-2000 | 2000 | 20 | 0 | 1 | 19 | 0.025 | 1 | 1 |

3 of 160 games hit the context limit; 3 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 3 of 160 games hit the context limit and were adjudicated (3 of them) instead of being scored as draws
