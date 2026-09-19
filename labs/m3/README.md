# M3 labs

Scripts referenced by the M3 lesson (`rukh-lab`, `curso/m3/01-el-encoder`). Run them from the
repository root.

| Script | Needs | Lesson lab |
|---|---|---|
| `bidirectional.py` | nothing | 1 · el encoder bidireccional y el enmascarado |
| `value_bar_export.py` | the heads checkpoint and Stockfish | visualización · `artifacts/web/value-bar.json` |

Both are copies of the code blocks in the lesson: if you edit one, edit the other. The only
deliberate difference is one wrapped line in `bidirectional.py` (the lesson prints the three
masking shares on a single line, which is 106 columns and this repository's ruff limit is 100);
the output is identical.

`artifacts/web/value-bar.json` is copied into the course with `pnpm sync:data`, like the two
JSON files of M2.

## `bidirectional.py`

Three checks, no checkpoint and no data — it builds its own toy encoder:

1. **It is not causal.** Change the last token of a sequence and the hidden state of the *first*
   one moves. This is the exact opposite of the assertion M2's `causal_mask.py` makes about the
   decoder, and running the two next to each other is the point.
2. **Padding does not leak.** The same three real tokens give the same hidden states whether or
   not `<pad>` travels with them, as long as the mask does.
3. **The 80/10/10 recipe is counted, not trusted**, over 10 000 tokens.

```bash
uv run python labs/m3/bidirectional.py
```

The `kept` share it prints is an upper bound on the real 10 %: a "random" replacement can draw
the token that was already there, and this counts that as kept.

## `value_bar_export.py`

Walks Byrne–Fischer (New York, 1956) up to 17...Be6, evaluates every position with the encoder's
heads **and** with Stockfish, and writes `artifacts/web/value-bar.json` (schema
`rukh-value-bar/1`, documented in `rukh-lab/src/data/README.md`) for the `ValueBar` island.

```bash
uv run python labs/m3/value_bar_export.py
```

It needs `checkpoints/encoder-heads-full/best.pt` and a Stockfish binary
(`scripts/get_stockfish.py`, or `RUKH_STOCKFISH`): both curves on one chart is the whole figure,
so it refuses to run with only one of them rather than writing half a file.
