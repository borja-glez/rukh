# Evaluation of `medium-v4-dpo-onpolicy-greedy`

- Suite: `full`
- Checkpoint: `E:/work/ai/chess-lm/rukh/checkpoints/medium-v4-dpo-onpolicy/dpo.pt`
- Weights SHA-256: `67eeee8a3712c7a6bb1e01eb1f879323ef15574e6ebc0a099e4dd8fd0e4bde33`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-21
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.2 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.2 % |
| Top-1 next move | 52.3 % |
| Top-3 next move | 79.9 % |
| Puzzles solved | 39.1 % |
| Estimated Elo | 1632 (95 % CI 1567-1706) |
| Mean centipawn loss | n/a |
| Opening diversity | 0.992 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 194 of 200 |
| Line entropy | 7.584 bits of 7.644 |
| Normalised | 0.992 |
| First-move entropy (no sampling) | 1.751 bits |

Most played lines:

| Line | Games |
|---|---:|
| `e2e4 d7d5 e4d5 d8d5 g1f3 c8g4 f1e2 b8c6 d2d4 e8c8 c1e3 e7e5` | 2 |
| `e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 e7e5 d4b5 d7d6 b1c3 a7a6` | 2 |
| `e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6` | 2 |
| `e2e4 d7d5 e4d5 d8d5 g1f3 c8g4 f1e2 b8c6 h2h3 g4f3 e2f3 d5e6` | 2 |
| `e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 d4c6 b7c6 f1d3 d7d5` | 2 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `docs/acceptance.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 992 | 99.2 % |
| sampled | 1000 | 992 | 99.2 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 48.3 % | 78.5 % |
| 2000-2200 | 328 | 54.9 % | 81.1 % |
| 2200-2400 | 97 | 59.8 % | 81.4 % |
| 2400-2600 | 28 | 67.9 % | 85.7 % |
| 2600+ | 4 | 100.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1180 | 59.0 % |
| 1500-2000 | 2000 | 806 | 40.3 % |
| 2000+ | 2000 | 357 | 17.8 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 14 | 3 | 3 | 0.775 | 2 | 2 |
| skill-0 | 1381 | 20 | 17 | 2 | 1 | 0.900 | 0 | 0 |
| skill-1 | 1467 | 20 | 13 | 0 | 7 | 0.650 | 0 | 0 |
| uci-1500 | 1500 | 20 | 11 | 0 | 9 | 0.550 | 1 | 1 |
| skill-2 | 1589 | 20 | 8 | 1 | 11 | 0.425 | 0 | 0 |
| skill-3 | 1678 | 20 | 11 | 1 | 8 | 0.575 | 0 | 0 |
| uci-1800 | 1800 | 20 | 8 | 1 | 11 | 0.425 | 0 | 0 |
| uci-2000 | 2000 | 20 | 2 | 2 | 16 | 0.150 | 1 | 1 |

4 of 160 games hit the context limit; 4 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 4 of 160 games hit the context limit and were adjudicated (4 of them) instead of being scored as draws
