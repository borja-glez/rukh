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

## Phase 1, module by module

Phase 1 (M0-M6, closed 2026-09-21) builds a chess language model from the raw Lichess database to
a measured, published, playable family of models. Everything below is reproducible with
`docs/reproducir.md`; every number lives in the
[single results table](https://lab.rukh.borjaglez.com/proyecto/) (`docs/benchmarks.md`), measured
the same day with the same suite by `rukh eval nightly`.

| Module | What it builds | Where it lands |
|---|---|---|
| M0 · Taller | the toolchain, Stockfish, MLflow, CI | this repo's `Quickstart` |
| M1 · Datos | 5.9 M games ≥ 1800 Elo in UCI, the 2 030-token vocabulary, puzzles and Stockfish labels | `chorcat/rukh-games-1800`, `rukh-tokenizer`, `rukh-puzzles-split`, `rukh-positions-eval` |
| M2 · Decoder | `tiny` (5 M), `small` (39 M) and `medium-v4` (115 M) trained from scratch; ONNX with parity | `chorcat/rukh-tiny`, `rukh-small`, `rukh-medium` |
| M3 · Encoder | masked-move pretraining and three heads (value, blunder, result) | `chorcat/rukh-encoder-mmm`, `rukh-encoder` |
| M4 · Afinado | Elo-conditioned and masters fine-tunes, two LoRA adapters, Qwen3 with QLoRA as a baseline | `chorcat/rukh-medium-elo`, `rukh-medium-masters`, `rukh-lora-e4`, `rukh-lora-d4`, `rukh-qwen3-pgn-qlora` |
| M5 · Alineamiento | a reward model, DPO on and off policy, GRPO against a verifiable reward | `chorcat/rukh-rm`, `rukh-medium-dpo`, `rukh-medium-grpo`, `rukh-pairs-dpo`, `rukh-pairs-onpolicy` |
| M6 · Evaluar | the table, the floor of the instrument, a public baseline, the cost of int8, cards and the collection | the table, the [`Rukh` collection](https://huggingface.co/collections/chorcat/rukh-6ab14873918eabbc3ee287b5), [rukh.borjaglez.com](https://rukh.borjaglez.com) |

Three things to know before reading the table:

- **Elo is a pair, model and sampling** (D-047): the same checkpoint at temperature 0.05 and at
  0.6 are two players about 180 Elo apart (1359 against 1181), and they are two rows.
- **The ladder repeats to sampling noise on a quiet machine** (D-137): the same model, same
  seed, four runs, 1538/1538 on the clock and 1541/1524 on a node budget. The interval of every
  Elo (about ±55) is wider than any of those gaps; the machine must be doing nothing else while
  a ladder runs.
- **Compare within one run of the table, never across days**: to measure a difference, play the
  difference (`rukh eval match`, M5).

## What is in here

```
src/rukh/            the `rukh` package and its `rukh` CLI
  cli.py             typer app: info · data · train · eval · play · export · publish · pull · engine · mlflow
  hub.py             `rukh pull`: the published artefacts, written where the configs read them
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

## Reproducing the course, module by module

Every module builds on what the previous one left on disk. `docs/reproducir.md` maps each module
to what it needs, the commands that produce it and what they took on the reference machine. The
two mechanisms behind it:

```sh
uv run rukh pull --list                 # every published model and dataset, and which module starts from it
uv run rukh pull --module m4            # everything M4 needs if you skipped M0-M3 (medium-v4, the games, ...)
uv run rukh pull medium-v4 rukh-pairs-dpo
```

- A pulled model lands under its **run name** (`checkpoints/medium-v4/best.pt`), which is how
  every config and lesson names it. A run you trained yourself lands in a stamped folder
  (`checkpoints/medium-v4-20260919-174623/`), and the same spelling finds it: a stable name that
  does not exist resolves to the newest stamped run of that name.
- The code is always `main`. The milestones are tagged (`git tag -n1 -l 'p*'`: `p0` to `p5`,
  plus `p3-elo-1200`) so the repository can be read as it was the day each one closed, but the
  tags are history, not starting points: `p2`, for instance, still carries the Elo ladder that
  D-070 later corrected.

### The encoder (M3)

`PositionEncoder` is the bidirectional half of the project: it reads a position and says how good
it is, whether the move that led to it threw the game away and how the game ends. It is pretrained
with masked move modeling on the `moves` scheme (the same UCI stream the decoder trains on) and
fine-tuned with three linear heads on the Stockfish labels of P1.

```sh
uv run rukh data evals                                         # the Stockfish scores the labels use
uv run rukh train encoder --config configs/train/encoder-mmm.yaml          # masked move modeling
uv run rukh train heads   --config configs/train/encoder-heads-moves.yaml  # heads on the MMM run
uv run rukh train heads   --config configs/train/encoder-heads.yaml --curve  # squares + 10/25/50/100 %
uv run rukh eval encoder  --model checkpoints/<run>/best.pt    # F1 vs the baseline, value, result
uv run rukh export --ckpt checkpoints/<run>/best.pt --kind encoder --out artifacts/onnx/encoder   --fp16 --int8                                                # `value` and `blunder` for the demo
uv run rukh encoder embed --positions data/labels/positions-labels.parquet --out embeddings.npy
```

The two schemes are the comparison of the module: `moves` feeds the encoder the line that reached
the position, `squares` feeds it the 69 tokens of the board itself. Only `moves` can start from a
masked-move checkpoint, and `rukh train heads` refuses a checkpoint trained on the other one
rather than reinterpreting its vocabulary. `--curve` writes the label-count curve into the
checkpoint, which is what fills the "labels needed" table of `rukh eval encoder`.

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
