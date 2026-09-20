---
license: apache-2.0
library_name: rukh
pipeline_tag: text-generation
language:
  - en
datasets:
  - chorcat/rukh-games-1800
  - chorcat/rukh-tokenizer
tags:
  - chess
  - rukh
  - decoder
  - onnx
---

# chorcat/rukh-tiny

A GPT decoder written from scratch that plays chess by predicting the next move of a game
written in UCI. This is the `tiny-greedy` stage of [Rukh](https://github.com/borja-glez/rukh), a course
that builds a chess language model end to end: 5,309,952 parameters, a
vocabulary of 2030 fixed tokens and a context of
200 moves.

Play against it in the browser: [https://rukh.borjaglez.com/?stage=tiny-int8](https://rukh.borjaglez.com/?stage=tiny-int8) · read how it was built:
[https://lab.rukh.borjaglez.com](https://lab.rukh.borjaglez.com)

## Results

Measured with `rukh eval --suite full` on 2026-09-20.

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 94.5 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 94.5 % |
| Top-1 next move | 40.3 % |
| Top-3 next move | 67.1 % |
| Puzzles solved | 8.9 % |
| Estimated Elo | 921 (95 % CI 713-1040) |

Puzzles by difficulty band:

| Band | Solved |
|---|---|
| 1000-1500 | 14.1 % |
| 1500-2000 | 8.6 % |
| 2000+ | 3.9 % |

Legality is measured **without** the legality mask, twice, because the two numbers answer
different questions:

- **argmax** is the share of validation positions whose single most likely token is a legal
  move, with no temperature and no top-k. It is a property of the weights and it is the
  definition behind the "at least 99 % legal" bar of the project.
- **sampled** draws the token exactly as the demo draws it, so it is what a player would meet
  with the mask switched off. It is always the lower of the two.

The demo masks illegal moves before sampling, so it never plays one.

### Acceptance bars

The project set these two bars for the decoder in `GOAL.md` before any of this was trained. This
is where the release stands against them, and it is the same table whether the answer is
flattering or not.

| Bar | Target | Measured | Verdict |
|---|---|---|---|
| Legality without the mask, argmax | at least 99 % | 94.5 % | **not met** |
| Estimated Elo | at least 1200 | 921 (95 % CI 713-1040) | **not met** |

**The legality bar is not met**: without the legality mask the weights propose an illegal move
more often than one time in a hundred. The demo masks before sampling and therefore never plays
one, but the bar is about the weights and the weights do not clear it.

**The Elo bar is not met.** This stage exists as a baseline, not as a release: it is the smallest
model in the project and it is published so the larger stages have something to be compared
against. The stages that carry the project's answer to this bar are `rukh-medium` (1504) and
`rukh-medium-dpo` (1529), trained on 19.0 M games where this one saw 2.9 M.

### The ratings on this card replace lower ones

An earlier release of these cards reported much lower ratings — 1007 for `small`, 1091 for
`medium`, 64 for `tiny` — and said the project's 1200 Elo bar was missed. **Those figures were
wrong, and the models were not.**

The rating is fitted against eight Stockfish opponents. Four of them are set with `Skill Level`
and were given Elo labels by hand, on the assumption that they reach below the engine's 1320
`UCI_Elo` floor. None of them does. Playing the ladder against itself — 40 games a pair, the
suite's own 0.1 s a move, colours alternated — put them at 1381, 1467, 1589 and 1678 against
labels of 800, 950, 1100 and 1250: between 428 and 581 Elo too low, every one in the same
direction, which dragged every stage's fit down by roughly 350 points.

The method checks itself: `uci-1500` measured **+179** Elo over `uci-1320` against a nominal
+180. (`UCI_Elo` does compress higher up — `uci-1800` measured +215 over `uci-1500`, not +300 —
which is a caveat for the top rungs, not for the range these models play in.)

Every stage was then re-evaluated on the measured ladder, and the numbers on this card come from
those runs. The correction moves all stages by a similar amount, so comparisons between them are
unchanged.

### An Elo belongs to the pair model+sampling

The same weights (`4591cf8cca0b`) measure **921 Elo** at
temperature 0.05, top-k 1 and **64 Elo** at temperature 0.6, top-k 20.
Nothing about the tensors changed between the two runs; only the way a move is drawn from them
did. A rating is therefore a property of the pair model+sampling and not of the file you
download, and any table that compares stages has to fix the sampling first.

The results above are the temperature 0.05, top-k 1 point. The demo samples for variety
rather than always playing its best move, so what a player meets over the board is the weaker of
the two numbers.

How to read these numbers:

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of GOAL.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 3 of 160 games hit the context limit and were adjudicated (3 of them) instead of being scored as draws


## Input and output

The model reads a game as a sequence of tokens and predicts the next one:

```
<bos> <w1800> <b1800> e2e4 e7e5 g1f3 ... <1-0> <eos>
```

The two header tokens are the 100-Elo bins of White and Black. Moves are UCI strings
(`e2e4`, `e7e8q`, and castling as the king's two-square move `e1g1`). The vocabulary is a fixed
enumeration, not learned from data, and ships in `tokenizer/vocab.json`.

## Files

- `model.safetensors`
- `config.json`
- `tokenizer/vocab.json`
- `onnx/model-fp16.onnx`
- `onnx/model-int8.onnx`
- `onnx/model.onnx`
- `onnx/parity.json`


The language-modelling head is tied to the token embedding, so the weights file holds
`tokens.weight` and **not** `lm_head.weight`: the two are one tensor, and storing it twice would
both break `safetensors` and double-count the embedding in the parameter count. Re-tie it after
loading (`model.lm_head.weight = model.tokens.weight`); `rukh` does it in the constructor.

The ONNX graph returns only the logits of the **last** step, `(batch, vocab)`: that is all the
demo needs and it keeps the output tensor two hundred times smaller. `model-fp16.onnx` is for
WebGPU and `model-int8.onnx` for the WASM fallback.

### How faithful the ONNX files are

Every exported file was run against the PyTorch checkpoint on 1000
validation positions, comparing the argmax move. `onnx/parity.json` in this
repository is that measurement, as the exporter wrote it.

| File | Same move as PyTorch | Worst logit drift |
|---|---|---|
| `model.onnx` (fp32) | 100.0 % | 4.39e-05 |
| `model-fp16.onnx` (fp16) | 99.8 % | 0.033 |
| `model-int8.onnx` (int8) | 96.1 % | 1.53 |

The bar the project set itself is 99.9 %.

`model-fp16.onnx` does not reach it: it picks a different move in 0.2 % of
positions, roughly one in 500.

`model-int8.onnx` does not reach it: it picks a different move in 3.9 % of
positions, roughly one in 26. That is the file the WASM fallback loads, so a phone on the
int8 build is playing a measurably different model from the one in the results table above: the
weights are the same, the arithmetic is not.

## Training recipe


| Parameter | Value |
|---|---|
| `batch_size` | `128` |
| `betas` | `[0.9, 0.95]` |
| `block` | `200` |
| `ckpt_every` | `1000` |
| `compile` | `True` |
| `data_manifest_sha` | `2274906c04342b8fc1632d3e1c3bff82e33f535936af742c7a7f9e4b052afa31` |
| `device` | `cuda` |
| `eval_batches` | `50` |
| `eval_every` | `250` |
| `grad_accum` | `2` |
| `grad_clip` | `1.0` |
| `log_every` | `10` |
| `lr` | `0.001` |
| `max_steps` | `6000` |
| `min_lr_ratio` | `0.1` |
| `model` | `None` |
| `num_params` | `5258752` |
| `out_dir` | `checkpoints` |
| `precision` | `bf16` |
| `preset` | `tiny` |
| `run_name` | `tiny` |
| `seed` | `42` |
| `tokens_dir` | `data/tokens/uci` |
| `unique_run_name` | `True` |
| `vocab_hash` | `527c5dda224cab570cb84da8f7dcda0f43fe53edaf86822f899568f5dd824b46` |
| `warmup` | `500` |
| `weight_decay` | `0.1` |
| `workers` | `4` |

MLflow run: `0a25dd056bfe4ca9b89204a6d75686f3`.

```json
{
  "architectures": [
    "MoveDecoder"
  ],
  "model_type": "rukh-move-decoder",
  "library_name": "rukh",
  "rukh_version": "0.0.1",
  "stage": "tiny-greedy",
  "step": 6000,
  "params": 5309952,
  "tokenizer": "uci",
  "vocab_hash": "527c5dda224cab570cb84da8f7dcda0f43fe53edaf86822f899568f5dd824b46",
  "data_manifest_sha": "2274906c04342b8fc1632d3e1c3bff82e33f535936af742c7a7f9e4b052afa31",
  "git_sha": "d2bb0e0ee60f90e2205e45d5df5603552da6d39c",
  "vocab_size": 2030,
  "n_layer": 6,
  "n_head": 4,
  "d_model": 256,
  "d_ff": null,
  "block": 200,
  "dropout": 0.0,
  "pos": "learned",
  "tie_embeddings": true
}
```

## Data

Trained on [`chorcat/rukh-games-1800`](https://huggingface.co/datasets/chorcat/rukh-games-1800), [`chorcat/rukh-tokenizer`](https://huggingface.co/datasets/chorcat/rukh-tokenizer),
derived from the [Lichess open database](https://database.lichess.org) (CC0): rated standard
games with both players at 1800+ Elo, at least 180 seconds of base time, 20 to 300 plies,
converted from SAN to legal UCI. Validation uses a month the model never saw.

## Limitations

- It is a move-sequence model, not a search engine: it has no lookahead and no evaluation
  function, so it blunders tactics that any engine sees instantly.
- It has no board state of its own. Given a position without its move history (a puzzle FEN,
  say) it plays from a much shorter prompt than it was trained on and is markedly weaker.
- Without the legality mask it proposes illegal moves at the rate in the table above. Any
  application must mask, as the demo does.
- It was trained on games between 1800+ humans on Lichess and imitates them, blunders included.
  It is not an oracle of good play and is not conditioned to be one.
- The estimated Elo is a fit to a few hundred games against a limited Stockfish, with a
  bootstrap interval: it is an estimate with a confidence interval, not a rating. The interval
  covers sampling noise only. The rungs below Stockfish's 1320 floor are nominal `Skill Level`
  anchors rather than measured ratings, and games that run out of context are adjudicated on the
  final position instead of being scored as draws.

## License

APACHE-2.0. The code and the weights are released under the Apache License 2.0;
the training data comes from Lichess under CC0. Please credit Lichess when you use them.

Generated with `rukh` 0.0.1.
