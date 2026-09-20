# Evaluation of `medium-v4-grpo-greedy`

- Suite: `full`
- Checkpoint: `checkpoints/medium-v4-grpo/grpo.pt`
- Weights SHA-256: `ade81558d491e041bc9fa784de6a213dd42c7d73920d1ecd12a200076df51c8c`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-20
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.3 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.3 % |
| Top-1 next move | 53.3 % |
| Top-3 next move | 80.3 % |
| Puzzles solved | 38.6 % |
| Estimated Elo | 1572 (95 % CI 1512-1640) |
| Mean centipawn loss | n/a |
| Opening diversity | 0.996 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 197 of 200 |
| Line entropy | 7.614 bits of 7.644 |
| Normalised | 0.996 |
| First-move entropy (no sampling) | 1.787 bits |

Most played lines:

| Line | Games |
|---|---:|
| `e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6` | 2 |
| `e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 d4c6 b7c6 f1d3 d7d5` | 2 |
| `e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6 f1d3 d8c7 e1g1 g8f6` | 2 |
| `d2d4 g8f6 g1f3 d7d5 c2c4 d5c4 b1c3 a7a6 a2a4 b8c6 e2e3 c6a5` | 1 |
| `e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 d2d4 c8g4 f1e2 e7e6 e1g1 c7c6` | 1 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `GOAL.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 993 | 99.3 % |
| sampled | 1000 | 993 | 99.3 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 49.9 % | 78.6 % |
| 2000-2200 | 328 | 55.2 % | 82.6 % |
| 2200-2400 | 97 | 60.8 % | 80.4 % |
| 2400-2600 | 28 | 67.9 % | 82.1 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1178 | 58.9 % |
| 1500-2000 | 2000 | 783 | 39.1 % |
| 2000+ | 2000 | 357 | 17.8 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 16 | 1 | 3 | 0.825 | 0 | 0 |
| skill-0 | 1381 | 20 | 12 | 2 | 6 | 0.650 | 2 | 2 |
| skill-1 | 1467 | 20 | 12 | 1 | 7 | 0.625 | 0 | 0 |
| uci-1500 | 1500 | 20 | 13 | 1 | 6 | 0.675 | 0 | 0 |
| skill-2 | 1589 | 20 | 9 | 0 | 11 | 0.450 | 0 | 0 |
| skill-3 | 1678 | 20 | 2 | 2 | 16 | 0.150 | 1 | 1 |
| uci-1800 | 1800 | 20 | 4 | 1 | 15 | 0.225 | 2 | 2 |
| uci-2000 | 2000 | 20 | 6 | 1 | 13 | 0.325 | 0 | 0 |

5 of 160 games hit the context limit; 5 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 5 of 160 games hit the context limit and were adjudicated (5 of them) instead of being scored as draws
