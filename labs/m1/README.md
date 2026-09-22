# M1 labs

Scripts referenced by the M1 lesson (`rukh-lab`, `curso/m1/10-labs-del-pipeline`). Run them from
the repository root after the pipeline steps they depend on:

| Script | Needs | Lesson lab |
|---|---|---|
| `explore.py` | `rukh data uci` | 1 · explorar el parquet con DuckDB |
| `loader_check.py` | `rukh data tokenize --scheme uci --pack` | 4 · dataloader y empaquetado |
| `bpe_merges.py` | `rukh data tokenize --scheme bpe` | 5 · fusiones del BPE |

The lesson quotes these files rather than copying them, and `pnpm verify:code` in `rukh-lab`
fails when a quoted block and its source drift apart, so editing one is enough.

## Starting point

M1 needs `data/raw/` from M0's `rukh data fetch`. To skip the fetch and the SAN-to-UCI conversion
of lab 2, `uv run rukh pull rukh-games-1800` puts the converted games in `data/uci/` directly;
`uv run rukh pull --module m1` also brings the tokenizer. The rest of the pipeline (`tokenize`,
`positions`, `evals`, `puzzles`, `pairs`, `elite`, `elo-bins`) runs from there, and every step,
with what it takes, is in `docs/reproducir.md`.
