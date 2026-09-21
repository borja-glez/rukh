# M4 labs

Scripts referenced by the M4 lesson (`rukh-lab`, `curso/m4/01-fine-tuning`). Run them from the
repository root.

| Script | Needs | Lesson lab |
|---|---|---|
| `header_histogram.py` | a packed token stream | 1 · mirar el corpus antes de culpar al modelo |
| `lora_check.py` | nothing (check 4 needs the `hf` extra) | 4 y 5 · LoRA a mano y la prueba contra `peft` |
| `games_needed.py` | a sweep's `results.json` | 6 · ¿faltan partidas o no hay diferencia? |
| `lora_spectrum.py` | a checkpoint and a trained adapter | la figura de la sección de LoRA |

Both are copies of the code the lesson shows, and both print the numbers the lesson quotes. If you
edit one, edit the other.

## Starting point

Everything in M4 starts from `checkpoints/medium-v4/best.pt` (M2, fourth part) and needs
`data/uci/`, the 44 months of `data/elite/games.parquet` and the tokenizer.
`uv run rukh pull --module m4` brings them, plus the five results of the module (`medium-elo`,
`medium-masters`, `lora-e4`, `lora-d4`, `qwen3-pgn-qlora`) for whoever only wants to measure
them. The club-level corpus and its flat Elo bins are rebuilt in minutes; the ordered commands,
with what each took, are `docs/runbooks/afinado-y-adaptadores.md`.

## `header_histogram.py`

Counts how many times the model has seen each Elo header. The second token of every packed game
*is* White's header, so the whole measurement is a `Counter` over `tokens[starts + 1]`.

On `data/tokens-v4/uci/train` — the corpus every published model trained on — it prints:

```
data/tokens-v4/uci/train: 18,942,740 games, 1,681,069,636 tokens

    <w1800>  1,570,327  #############
    ...
    <w2500>  5,969,302  ################################################
    ...
headers below <w1800>: 0 of 12 ever seen in this corpus
```

Zero of twelve. That is the whole of M4's first section: the headers below 1800 exist in the
vocabulary, have a row in the embedding table, and never received a gradient. Run it again with
`--tokens data/tokens-elo/uci/train` to see the corpus that fixed it.

## `lora_check.py`

Four claims about the hand-written LoRA, in order of how easy they are to get wrong, on a toy
decoder it builds itself. The fourth is the one that makes the chapter honest: the same
configuration trained with `apply_lora` and with `peft.get_peft_model`, on the same base weights
and with the same `A` copied across, has to optimise the same number. It does, bit for bit:

```
4. step   ours          peft          |delta|
      0   4.17093372    4.17093372    0.00e+00
      ...
      5   4.15640736    4.15640736    0.00e+00
```

The comparison runs on `attn.proj` and not on the query and value projections, and that is not a
detail: `peft` cannot express what we do on a fused `qkv`. Asked for `target_modules=["qkv"]` it
adapts all three projections with **one** `A`/`B` pair of 2304 rows, while `apply_lora` gives one
pair per slice. Those are different parameterisations, and comparing them would measure the
difference rather than the correctness of the code.

## `games_needed.py`

The question every overlapping interval raises, answered with arithmetic instead of machine time.
The ladder scores a proportion, so its standard error is `sqrt(p (1 - p) / n)` and two 95 %
intervals stop touching when the gap between the scores beats `1.96 (se1 + se2)` — about
`n > 3.84 / (delta p)^2` games per condition for proportions near a half.

On the sweep this milestone ran, with `--only 1500,2000,2400` (the conditions `GOAL.md` names):

```
  <w1500> -> <w2000>  0.466->0.453  -0.013   no              24,420
  <w2000> -> <w2400>  0.453->0.569  +0.116   no                 284
```

Twenty-four thousand games is sixty hours of Stockfish to tighten an interval around a difference
that is not there; two hundred and eighty-four is an hour. The two rows read the same on the
report — "not separated" — and mean opposite things, which is the whole point of running this
before deciding whether to re-run anything.

## `lora_spectrum.py`

"The correction has to pass through eight dimensions" is easy to write and easy to believe without
checking. The check is one line of linear algebra: the singular values of `W` and of
`delta W = (alpha / r) B A` for the same matrix.

On the published `lora-e4`, over the query slice of `blocks.0.attn.qkv`:

```
  r = 8, alpha = 16, scale = 2.0
  non-zero singular values: W 768, delta W 8
  ||delta W||_F / ||W||_F = 0.0441
  first singular values of delta W: [0.8338, 0.4277, 0.3592, 0.3211, 0.2811, 0.2302, 0.1991, 0.1774, 0.0, 0.0]
```

Eight, and the ninth is `0.0` rather than something small. It writes
`artifacts/web/lora-spectrum.json`, which is what the lesson's `LoraSpectrum` figure draws.
