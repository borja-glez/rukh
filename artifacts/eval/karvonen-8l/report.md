# Evaluation of `karvonen-8l` (public baseline)

- Source: `adamkarvonen/chess_llms/lichess_8layers_ckpt_no_optimizer.pt`
- Checkpoint: `E:/work/ai/chess-lm/rukh/checkpoints/karvonen-8l/lichess_8layers_ckpt_no_optimizer.pt`
- Weights SHA-256: `84076e42e006c8edeabcb593e0b67469bdea17e1beabe78a459c2dd1f22c0668`
- Parameters: 25,714,688
- Device: `cuda:0`
- Date: 2026-09-21

A character-level nanoGPT trained by Adam Karvonen on 16 M Lichess games written as `;1.e4 e5 2.Nf3`, measured here with the same positions, puzzles and Stockfish ladder as every other row. It was not trained by this course and nothing about it was tuned.

## Headline

| Metric | Value |
|---|---|
| Legal moves written, no mask | 99.60 % |
| Top-1 next move | 50.7 % |
| Puzzles solved | 28.1 % |
| Estimated Elo | 1328 (95 % CI 1250-1391) |

## How the answers failed

| Outcome | Count | Share |
|---|---:|---:|
| legal | 996 | 99.60 % |
| illegal here | 4 | 0.40 % |
| not a move at all | 0 | 0.00 % |
| ambiguous SAN | 0 | 0.00 % |
| nothing written | 0 | 0.00 % |
| **asked** | **1000** | |

## Puzzles by band

Prompts answered without a game prefix: 0.

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 950 | 47.5 % |
| 1500-2000 | 2000 | 520 | 26.0 % |
| 2000+ | 2000 | 216 | 10.8 % |

## Games against Stockfish

| Rung | Games | Score |
|---|---:|---:|
| uci-1320 | 20 | 0.575 |
| skill-0 | 20 | 0.450 |
| skill-1 | 20 | 0.200 |
| uci-1500 | 20 | 0.275 |
| skill-2 | 20 | 0.250 |
| skill-3 | 20 | 0.100 |
| uci-1800 | 20 | 0.000 |
| uci-2000 | 20 | 0.050 |

## Notes

- this model has no rating header: header_elo does not condition this row
