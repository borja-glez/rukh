# M1 labs

Scripts referenced by the M1 lesson (`rukh-lab`, `curso/m1/01-datos-y-tokenizacion`). Run them from
the repository root after the pipeline steps they depend on:

| Script | Needs | Lesson lab |
|---|---|---|
| `explore.py` | `rukh data uci` | 1 · explorar el parquet con DuckDB |
| `loader_check.py` | `rukh data tokenize --scheme uci --pack` | 4 · dataloader y empaquetado |
| `bpe_merges.py` | `rukh data tokenize --scheme bpe` | 5 · fusiones del BPE |

They are kept in sync by hand with the code blocks in the lesson: if you edit one, edit the other.
