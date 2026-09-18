# rukh

Rukh is a chess language model built from scratch, and the machine-learning half of a course on
generative and agentic AI. This repository holds the data pipeline, the models, the training and
evaluation code, the export to ONNX and, in phase 2, the agents. The course lives in
[`rukh-lab`](https://github.com/borja-glez/rukh-lab) (published at
[lab.rukh.borjaglez.com](https://lab.rukh.borjaglez.com)) and the playable demo in
[`rukh-web`](https://github.com/borja-glez/rukh-web)
([rukh.borjaglez.com](https://rukh.borjaglez.com)). Everything trained is published on Hugging
Face under [`chorcat`](https://huggingface.co/chorcat) as `rukh-*`.

The name is Persian for the rook.

## What is in here

```
src/rukh/            the `rukh` package and its `rukh` CLI
  cli.py             typer app: info · data fetch · engine check · mlflow ui
  env.py             environment report (Python, torch, CUDA, GPU, Stockfish, MLflow, dirs)
  paths.py           project directories; RUKH_HOME overrides the root
  config.py          strict pydantic configs (extra="forbid") loaded from YAML
  data/              fetch.py (DuckDB over hf:// parquet) · manifest.py (provenance record)
  engine.py          Stockfish detection and a short UCI_Elo sanity game
  tracking.py        local MLflow store (SQLite) and run helper
configs/             YAML configs (configs/data/lichess-2025-01-02.yaml)
scripts/             get_stockfish.py (downloads Stockfish 19 into tools/, gitignored)
tests/unit/          fast CPU tests (`pytest -m unit`); `engine` tests need Stockfish
labs/                thin notebooks behind each lesson; export JSON to artifacts/web/
artifacts/web/       small JSON the course site renders (versioned)
gold/                golden files for evaluation
docs/spec/           the binding design docs (Spanish); docs/decisiones-de-ejecucion.md is the
                     decision ledger; adr/, runbooks/, model-cards/, benchmarks.md, backlog.md
data/ mlruns/ tools/ local datasets, MLflow store and binaries (gitignored)
```

## Quickstart

Requirements: Python 3.12 managed by [`uv`](https://docs.astral.sh/uv/) (the repo pins it in
`.python-version`), and an NVIDIA GPU with CUDA 12.8 for the `cu128` extra. The `cpu` extra is what
CI uses; the two extras are mutually exclusive and lock the same `torch>=2.11,<2.12` range.

```sh
uv sync --extra cu128 --group dev      # or --extra cpu on a machine without a GPU
uv run rukh info                       # Python, torch, CUDA, GPU, Stockfish, MLflow store
uv run python scripts/get_stockfish.py # Stockfish 19 into tools/stockfish/ (idempotent)
uv run rukh engine check               # plays 40 plies at UCI_Elo 1400 against a random mover
uv run rukh data fetch --config configs/data/lichess-2025-01-02.yaml --dry-run
uv run rukh mlflow ui                  # MLflow UI on the local SQLite store
uv run pytest -m unit -q               # fast tests; add `or engine` once Stockfish is installed
```

`rukh info --json`, `rukh engine check --json` and `rukh data fetch --json` print machine-readable
output. Set `RUKH_HOME` to move `data/`, `mlruns/` and `tools/` elsewhere, and `RUKH_STOCKFISH` to
point at a specific binary.

## Development

- `uv run ruff check .` and `uv run ruff format .` must be clean; CI runs them plus
  `pytest -m unit` on CPU (`.github/workflows/ml.yml`).
- Configs are pydantic models with `extra="forbid"`: an unknown key is an error.
- Test markers: `unit` (fast, CPU, no network), `engine` (needs Stockfish), `gpu`, `slow`.
- Commits follow Conventional Commits in English; the design docs and the course are in Spanish.

## Licenses and attribution

- Code: [MIT](LICENSE).
- Model weights published on Hugging Face: Apache-2.0.
- Data: derived from the [Lichess open database](https://database.lichess.org/) and its mirrors
  on Hugging Face (`Lichess/standard-chess-games`, `Lichess/chess-puzzles`,
  `Lichess/chess-position-evaluations`, `Lichess/chess-openings`), released under CC0 1.0. The
  derived datasets are published under CC0 with attribution to Lichess.
- Stockfish (GPL-3) and python-chess (GPL-3) are used as external tools/libraries on the training
  side only; neither is redistributed with the web demo.
- Course content (in `rukh-lab`): CC BY-NC-SA 4.0.
