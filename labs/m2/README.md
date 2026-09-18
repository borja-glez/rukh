# M2 labs

Scripts referenced by the M2 lesson (`rukh-lab`, `curso/m2/01-el-decoder`). Run them from the
repository root.

| Script | Needs | Lesson lab |
|---|---|---|
| `params.py` | nothing | 1 · contar los parámetros a mano |
| `causal_mask.py` | nothing | 1 · comprobar la causalidad |
| `attention_export.py` | a trained checkpoint | visualización · `artifacts/web/attention.json` |
| `replay_export.py` | an MLflow run (`rukh train`) | visualización · `artifacts/web/training-replay.json` |

`params.py`, `causal_mask.py` and `attention_export.py` are copies of the code blocks in the
lesson: if you edit one, edit the other. `replay_export.py` is only described there, so this file
is its source of truth.

Both JSON outputs are copied into the course with `pnpm sync:data`.
