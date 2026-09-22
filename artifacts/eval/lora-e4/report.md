# Evaluation of `lora-e4`

- Suite: `full`
- Checkpoint: `E:/work/ai/chess-lm/rukh/artifacts/eval/merged/lora-e4.pt`
- Weights SHA-256: `be7ee02f65ff12cca30313c2570c02b217d984e7a7a8c6ef40776cff520f3298`
- Parameters: 115,120,128
- Device: `cuda`
- Date: 2026-09-21
- MLflow run: not tracked

## Headline

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.8 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.8 % |
| Top-1 next move | 54.7 % |
| Top-3 next move | 83.3 % |
| Puzzles solved | 37.8 % |
| Estimated Elo | 1538 (95 % CI 1488-1604) |
| Mean centipawn loss | n/a |
| Opening diversity | 0.999 |

## Opening diversity

200 self-play openings of 12 plies, drawn at temperature 1.0 with top-k 20. **Not** the suite's sampling: at the near-deterministic setting every stage plays one single opening and scores 0, which measures the sampler and not the weights (D-047).

| Metric | Value |
|---|---:|
| Distinct opening lines | 199 of 200 |
| Line entropy | 7.634 bits of 7.644 |
| Normalised | 0.999 |
| First-move entropy (no sampling) | 0.021 bits |

Most played lines:

| Line | Games |
|---|---:|
| `e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 c5d4 c3d4 f8b4` | 2 |
| `e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f8c5 c2c3 a7a6 b5a4 b7b5` | 1 |
| `e2e4 d7d5 e4d5 g8f6 f1c4 f6d5 g1f3 c8g4 e1g1 e7e6 d2d3 c7c6` | 1 |
| `e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1c4 e7e6` | 1 |
| `e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5` | 1 |

## Legality

Two rates, because they answer different questions. **argmax** is the share of validation positions whose single most likely token is a legal move, with no temperature, no top-k and no mask: it is a property of the weights and it is the definition behind the ≥ 99 % bar of `docs/acceptance.md`. **sampled** draws the token exactly as the demo does(T=0.05, top-k 1), so it is what a player would meet with the mask switched off, and it is always the lower of the two.

| Definition | Positions | Legal | Rate |
|---|---:|---:|---:|
| argmax | 1000 | 998 | 99.8 % |
| sampled | 1000 | 998 | 99.8 % |

## Next-move accuracy by Elo band

| Band | Positions | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1800-2000 | 543 | 51.9 % | 82.1 % |
| 2000-2200 | 328 | 56.4 % | 84.5 % |
| 2200-2400 | 97 | 60.8 % | 84.5 % |
| 2400-2600 | 28 | 64.3 % | 85.7 % |
| 2600+ | 4 | 75.0 % | 100.0 % |

## Puzzles by difficulty band

Prompt: game-prefix.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 1162 | 58.1 % |
| 1500-2000 | 2000 | 764 | 38.2 % |
| 2000+ | 2000 | 343 | 17.2 % |

## Games against Stockfish

| Rung | Opponent Elo | Games | W | D | L | Score | Cut | Adjudicated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uci-1320 | 1320 | 20 | 15 | 1 | 4 | 0.775 | 0 | 0 |
| skill-0 | 1381 | 20 | 11 | 2 | 7 | 0.600 | 2 | 2 |
| skill-1 | 1467 | 20 | 15 | 2 | 3 | 0.800 | 0 | 0 |
| uci-1500 | 1500 | 20 | 10 | 2 | 8 | 0.550 | 0 | 0 |
| skill-2 | 1589 | 20 | 7 | 2 | 11 | 0.400 | 1 | 1 |
| skill-3 | 1678 | 20 | 3 | 1 | 16 | 0.175 | 0 | 0 |
| uci-1800 | 1800 | 20 | 3 | 2 | 15 | 0.200 | 2 | 2 |
| uci-2000 | 2000 | 20 | 0 | 5 | 15 | 0.125 | 0 | 0 |

5 of 160 games hit the context limit; 5 of those were adjudicated on the final position (shallow engine analysis, or the material count when no engine was available) rather than scored as draws.

## Notes

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 5 of 160 games hit the context limit and were adjudicated (5 of them) instead of being scored as draws
