# 09 · Referencias (verificadas el 2026-09-18)

## Datos

- Lichess open database (CC0): https://database.lichess.org/ — partidas mensuales `.pgn.zst`, puzles, evaluaciones.
- Hugging Face, organización Lichess (todo CC0 salvo `tournament-chess-games`, CC BY-SA 4.0):
  - https://huggingface.co/datasets/Lichess/standard-chess-games — 6 771 826 271 partidas; parquet por `year=/month=`; un mes reciente ≈ 73 GB en 72 ficheros; columnas `Event, Site, White, Black, Result, WhiteTitle, BlackTitle, WhiteElo, BlackElo, WhiteRatingDiff, BlackRatingDiff, UTCDate, UTCTime, ECO, Opening, Termination, TimeControl, movetext`.
  - https://huggingface.co/datasets/Lichess/chess-puzzles — 6 100 960 puzles (877 MB), actualizado 2026-09-07.
  - https://huggingface.co/datasets/Lichess/chess-position-evaluations — 394 669 566 posiciones con `fen, line, depth, knodes, cp, mate`; 20 parquet ≈ 42 GB; actualizado 2026-07-08.
  - https://huggingface.co/datasets/Lichess/chess-openings — 3 704 aperturas ECO con `pgn`, `uci`, `epd`.
  - https://huggingface.co/datasets/Lichess/chess-puzzles-with-games, https://huggingface.co/datasets/Lichess/fishnet-evals (10-100 B filas; no se usa).
- Lichess Elite Database (nikonoel): https://database.nikonoel.fr/ — 2500+ contra 2300+ (desde 2021-12), sin bullet, zip mensuales de 60-100 MB.
- Lichess openings (repo fuente): https://github.com/lichess-org/chess-openings
- Project Gutenberg: Capablanca, *Chess Fundamentals* https://www.gutenberg.org/ebooks/33870 · Edward Lasker, *Chess Strategy* https://www.gutenberg.org/ebooks/5614 · tema "Chess": https://www.gutenberg.org/ebooks/subject/1677

## Trabajos previos (calibración y baselines)

- Adam Karvonen, *Chess-GPT's Internal World Model* (2024): https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html — nanoGPT 50M sobre 16M partidas (PGN a nivel de carácter), ~1300 Elo, 99,8 % legales en un día. Código: https://github.com/adamkarvonen/chess_llm_interpretability · modelos: https://huggingface.co/adamkarvonen/chess_llms (MIT) · datos: https://huggingface.co/datasets/adamkarvonen/chess_games
- Google DeepMind, *Amortized Planning with Large-Scale Transformers: A Case Study on Chess* (NeurIPS 2024) / *Grandmaster-Level Chess Without Search*: https://github.com/google-deepmind/searchless_chess — ChessBench (10M partidas, 15 000 M anotaciones de Stockfish 16), modelos 9M/136M/270M, 2895 Elo blitz. https://arxiv.org/abs/2402.04494
- Maia-2 (NeurIPS 2024) y Maia-3: https://github.com/CSSLab/maia2 · https://github.com/CSSLab/maia3 — predicción de jugada humana por nivel de Elo.
- *Can Large Language Models Develop Strategic Reasoning? Post-training Insights from Learning Chess* (2025): https://arxiv.org/abs/2507.00726 — RL con recompensas densas de una red de valor de ajedrez; mejora sobre recompensas binarias pero techo bajo si el modelo base no entiende ajedrez (argumento a favor de preentrenar el decoder propio).
- Lichess bots: https://lichess.org/@/lichess/blog/welcome-lichess-bots/WvDNticA · https://github.com/lichess-bot-devs/lichess-bot (AGPL-3).

## Herramientas y versiones (PyPI / npm, 2026-09-18)

| Paquete | Versión | Licencia |
|---|---|---|
| python-chess | 1.999 | GPL-3.0+ |
| torch (índice cu128, cp312 win) | 2.11.0+cu128 | BSD |
| transformers | 5.17.0 | Apache-2.0 |
| trl (SFTTrainer, DPOTrainer, GRPOTrainer, RewardTrainer, GKDTrainer) | 1.13.0 | Apache-2.0 |
| peft | 0.21.0 | Apache-2.0 |
| datasets | 5.0.1 | Apache-2.0 |
| accelerate | 1.15.0 | Apache-2.0 |
| onnx / onnxruntime | 1.22.0 / 1.30.0 | Apache-2.0 / MIT |
| mlflow | 3.16.1 | Apache-2.0 |
| fastapi | 0.141.1 | MIT |
| duckdb / polars / pyarrow / zstandard | 1.5.5 / 1.44.2 / 25.0.1 / 0.25.0 | MIT / MIT / Apache-2.0 / BSD |
| langchain / langchain-core / langgraph / langsmith | 1.4.2 / 1.6.3 / 1.2.11 / 0.13.0 | MIT |
| mcp / langchain-mcp-adapters | 2.2.0 / 0.3.2 | MIT |
| ragas / chromadb / faiss-cpu / sentence-transformers | 0.4.3 / 1.5.9 / 1.15.1 / 6.1.0 | Apache-2.0 / Apache-2.0 / MIT / Apache-2.0 |
| ollama / openai (clientes) | 0.6.2 / 3.16.1 | MIT / Apache-2.0 |
| chess.js (npm) | 1.4.0 | BSD-2-Clause |
| cm-chessboard (npm) | 8.14.0 | MIT |
| chessground (npm) | 9.2.1 | GPL-3.0-or-later (no se usa) |
| stockfish (npm, WASM) | 19.0.0 | GPL-3.0 (no se usa en fase 1) |
| onnxruntime-web (npm) | 1.30.0 | MIT |
| Astro | ≥ 7.3.2 | MIT |

Stockfish (binarios): https://stockfishchess.org/download/ (GPL-3). TRL docs: https://huggingface.co/docs/trl/index

## Diseño

- Portfolio de referencia: `E:\work\borjaglez.com` (Astro 7.3.2, `src/styles/global.css` con tokens `light-dark()`, `src/components/{Header,Footer,Rail,SectionHead,FigureFrame}.astro`, `src/layouts/Base.astro`, GSAP).
