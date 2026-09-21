# Evaluation of `qwen3-pgn-qlora`

- Base model: `Qwen/Qwen3-0.6B`
- Adapter: `E:/work/ai/chess-lm/rukh/checkpoints/qwen3-pgn-qlora`
- Parameters: 596,049,920
- Device: `cuda:0`
- Date: 2026-09-21

## Headline

| Metric | Value |
|---|---|
| Legal moves written, no mask | 62.50 % |
| Top-1 next move | 12.5 % |
| Puzzles solved | 0.9 % |
| Estimated Elo | < 807 (one-sided 95 % bound; every game lost) |

## How the answers failed

A decoder cannot write a move that does not exist: its vocabulary is the set of moves, so its only failure is a legal move in the wrong position. A model writing SAN has three more ways to be wrong, and folding them together would hide what the representation costs.

| Outcome | Count | Share |
|---|---:|---:|
| legal | 625 | 62.50 % |
| illegal here | 337 | 33.70 % |
| not a move at all | 29 | 2.90 % |
| ambiguous SAN | 9 | 0.90 % |
| nothing written | 0 | 0.00 % |
| **asked** | **1000** | |

## Puzzles by band

| Band | Attempted | Solved | Rate |
|---|---:|---:|---:|
| 1000-1500 | 2000 | 35 | 1.8 % |
| 1500-2000 | 2000 | 13 | 0.7 % |
| 2000+ | 2000 | 8 | 0.4 % |

## Games against Stockfish

| Rung | Games | Score |
|---|---:|---:|
| uci-1320 | 20 | 0.000 |
| skill-0 | 20 | 0.000 |
| skill-1 | 20 | 0.000 |
| uci-1500 | 20 | 0.000 |
| skill-2 | 20 | 0.000 |
| skill-3 | 20 | 0.000 |
| uci-1800 | 20 | 0.000 |
| uci-2000 | 20 | 0.000 |
