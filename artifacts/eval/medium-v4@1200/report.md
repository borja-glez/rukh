# Evaluation of `medium-v4@1200`

- Suite: `full`
- Checkpoint: `checkpoints/medium-v4-20260919-174623/best.pt`
- Weights SHA-256: `98e05fcdfa2d1619ba33294e2d7f775c677b898e947b570ec2633de0e824b85e`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-20
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.6 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.6 % |
| Top-1 next move | 49.8 % |
| Top-3 next move | 78.6 % |
| Puzzles solved | 36.4 % |
| Estimated Elo | 1419 (95 % CI 1358-1479) |
| Mean centipawn loss | n/a |
| Opening diversity | 1.000 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 200 of 200 |
| Line entropy | 7.644 bits of 7.644 |
| Normalised | 1.000 |
| First-move entropy (no sampling) | 2.856 bits |

Most played lines:

| Line | Games |
|---|---:|
| `e2e4 b8c6 g1f3 c6b4 d2d4 b4a2 a1a2 g8f6 e4e5 f6d5 c2c4 d5b4` | 1 |
| `e2e4 d7d5 e4d5 g8f6 g1f3 f6d5 f1e2 b8c6 e1g1 c8f5 d2d3 e7e5` | 1 |
| `d2d4 g8f6 g1f3 d7d5 c2c4 e7e6 e2e3 d5c4 f1c4 a7a6 e1g1 b8d7` | 1 |
| `g1f3 a7a6 g2g3 g8f6 f1g2 h7h6 e1g1 c7c5 f3e5 d8c7 d2d4 c5d4` | 1 |
| `f2f4 e7e6 e2e3 d8h4 g2g3 h4h6 f1g2 h6f6 g1f3 g8h6 d1e2 h6f5` | 1 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `docs/acceptance.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 996 | 99.6 % |
| sampled | 1000 | 996 | 99.6 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| <1800 | 1000 | 49.8 % | 78.6 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1120 | 56.0 % |
| 1500-2000 | 2000 | 741 | 37.0 % |
| 2000+ | 2000 | 324 | 16.2 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 12 | 3 | 5 | 0.675 | 1 | 1 |
| skill-0 | 1381 | 20 | 11 | 2 | 7 | 0.600 | 0 | 0 |
| skill-1 | 1467 | 20 | 5 | 0 | 15 | 0.250 | 0 | 0 |
| uci-1500 | 1500 | 20 | 7 | 0 | 13 | 0.350 | 0 | 0 |
| skill-2 | 1589 | 20 | 6 | 2 | 12 | 0.350 | 0 | 0 |
| skill-3 | 1678 | 20 | 2 | 2 | 16 | 0.150 | 1 | 1 |
| uci-1800 | 1800 | 20 | 0 | 0 | 20 | 0.000 | 1 | 1 |
| uci-2000 | 2000 | 20 | 3 | 3 | 14 | 0.225 | 0 | 0 |

3 of 160 games hit the context limit; 3 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 3 of 160 games hit the context limit and were adjudicated (3 of them) instead of being scored as draws
