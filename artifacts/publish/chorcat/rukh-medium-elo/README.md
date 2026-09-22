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

# chorcat/rukh-medium-elo

A GPT decoder written from scratch that plays chess by predicting the next move of a game
written in UCI. This is the `medium-elo` stage of [Rukh](https://github.com/borja-glez/rukh), a course
that builds a chess language model end to end: 115,120,128 parameters, a
vocabulary of 2030 fixed tokens and a context of
200 moves.

Play against it in the browser: [https://rukh.borjaglez.com/?stage=medium-elo-fp16](https://rukh.borjaglez.com/?stage=medium-elo-fp16) · read how it was built:
[https://lab.rukh.borjaglez.com](https://lab.rukh.borjaglez.com)

## Results

Measured with `rukh eval --suite full` on 2026-09-21.

| Metric | Value |
|---|---|
| Legality without the mask, argmax | 99.7 % |
| Legality without the mask, sampled (T=0.05, top-k 1) | 99.7 % |
| Top-1 next move | 54.9 % |
| Top-3 next move | 82.5 % |
| Puzzles solved | 37.8 % |
| Estimated Elo | 1592 (95 % CI 1532-1650) |


## What the Elo header does

The two header tokens `<wXXXX> <bXXXX>` are an instruction: *play like somebody of this strength*.
Below is the same suite run once per condition -- same opponents, same positions, same puzzles,
same seed -- with only those two tokens changed.

| Asked for | Elo | 95 % CI | legal argmax | top-1 | first-move entropy |
|---|---:|---|---:|---:|---:|
| `<w1200>` | 1425 | 1361-1479 | 100.0 % | 51.4 % | 1.596 bits |
| `<w1500>` | 1549 | 1481-1609 | 100.0 % | 53.9 % | 1.647 bits |
| `<w1800>` | 1498 | 1435-1552 | 99.8 % | 54.1 % | 1.741 bits |
| `<w2000>` | 1538 | 1479-1597 | 99.7 % | 54.0 % | 1.841 bits |
| `<w2100>` | 1606 | 1543-1670 | 99.7 % | 53.4 % | 1.881 bits |
| `<w2400>` | 1644 | 1578-1707 | 99.7 % | 53.3 % | 1.992 bits |

Point estimates are **not monotonic** and the intervals are **not separated**, with 219 Elo between the weakest and the strongest condition.

Read it for what it says. The **entropy of the first move** -- the model's own distribution over
the twenty legal openings, with no sampling and no seed -- rises in order with the condition, which
is the header controlling the **repertoire**. The **Elo** does not separate, and the same sweep run
on the model *before* this fine-tune covers most of the same range using headers it never trained
on, so the Elo column cannot tell conditioning from a prefix simply getting in the way. This model
changes **how it plays**, not **how well**.

Puzzles by difficulty band:

| Band | Solved |
|---|---|
| 1000-1500 | 59.8 % |
| 1500-2000 | 37.1 % |
| 2000+ | 16.6 % |

Legality is measured **without** the legality mask, twice, because the two numbers answer
different questions:

- **argmax** is the share of validation positions whose single most likely token is a legal
  move, with no temperature and no top-k. It is a property of the weights and it is the
  definition behind the "at least 99 % legal" bar of the project.
- **sampled** draws the token exactly as the demo draws it, so it is what a player would meet
  with the mask switched off. It is always the lower of the two.

The demo masks illegal moves before sampling, so it never plays one.

### Acceptance bars

The project set these two bars for the decoder in `docs/acceptance.md` before any of this was trained. This
is where the release stands against them, and it is the same table whether the answer is
flattering or not.

| Bar | Target | Measured | Verdict |
|---|---|---|---|
| Legality without the mask, argmax | at least 99 % | 99.7 % | met |
| Estimated Elo | at least 1200 | 1592 (95 % CI 1532-1650) | met |

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

How to read these numbers:

- legality_argmax is the share of validation positions where the single most likely token is a legal move (no temperature, no top-k, no mask): this is the >= 99 % bar of docs/acceptance.md. legality_sampled draws the token the way the demo does (temperature 0.05, top-k 1) and is always the lower of the two.
- the Elo interval covers sampling noise only: the four ``skill-*`` rungs are nominal ``Skill Level`` anchors rather than measured ratings, and Stockfish plays at 0.1 s per move, far below any setting ``UCI_Elo`` is calibrated for
- 2 of 160 games hit the context limit and were adjudicated (2 of them) instead of being scored as draws


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
| `model.onnx` (fp32) | 100.0 % | 2.43e-05 |
| `model-fp16.onnx` (fp16) | 99.9 % | 0.0119 |
| `model-int8.onnx` (int8) | 96.4 % | 2.16 |

The bar the project set itself is 99.9 %.

`model-int8.onnx` does not reach it: it picks a different move in 3.6 % of
positions, roughly one in 28. That is the file the WASM fallback loads, so a phone on the
int8 build is playing a measurably different model from the one in the results table above: the
weights are the same, the arithmetic is not.

## Training recipe


| Parameter | Value |
|---|---|
| `batch_size` | `32` |
| `betas` | `[0.9, 0.95]` |
| `block` | `200` |
| `ckpt_every` | `500` |
| `compile` | `True` |
| `data_manifest_sha` | `aa0809618dbc372d3cc192fef9dd110dca1ce4aae22dca13ea648c02d78c10b4` |
| `device` | `cuda` |
| `eval_batches` | `50` |
| `eval_every` | `200` |
| `grad_accum` | `8` |
| `grad_clip` | `1.0` |
| `init_from` | `E:/work/ai/chess-lm/rukh/checkpoints/medium-v4-20260919-174623/best.pt` |
| `log_every` | `10` |
| `lr` | `0.0001` |
| `max_steps` | `3800` |
| `min_lr_ratio` | `0.1` |
| `model` | `None` |
| `num_params` | `114966528` |
| `out_dir` | `checkpoints` |
| `precision` | `bf16` |
| `preset` | `medium` |
| `run_name` | `medium-elo` |
| `seed` | `42` |
| `tokens_dir` | `data/tokens-elo/uci` |
| `unique_run_name` | `True` |
| `vocab_hash` | `527c5dda224cab570cb84da8f7dcda0f43fe53edaf86822f899568f5dd824b46` |
| `warmup` | `200` |
| `weight_decay` | `0.1` |
| `workers` | `4` |

MLflow run: `326dc2926bfc409fbc529acdb7aaa1ef`.

```json
{
  "architectures": [
    "MoveDecoder"
  ],
  "model_type": "rukh-move-decoder",
  "library_name": "rukh",
  "rukh_version": "0.0.1",
  "stage": "medium-elo",
  "step": 3800,
  "params": 115120128,
  "tokenizer": "uci",
  "vocab_hash": "527c5dda224cab570cb84da8f7dcda0f43fe53edaf86822f899568f5dd824b46",
  "data_manifest_sha": "aa0809618dbc372d3cc192fef9dd110dca1ce4aae22dca13ea648c02d78c10b4",
  "git_sha": "c58affdff7c339792a0908b0b375ae82995701ef",
  "vocab_size": 2030,
  "n_layer": 16,
  "n_head": 12,
  "d_model": 768,
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

## In the course

- Built in [M4 · Los labs del afinado](https://lab.rukh.borjaglez.com/curso/m4/10-labs-de-afinado/), the lesson that runs every command behind this repository.
- Measured in the [single results table](https://lab.rukh.borjaglez.com/proyecto/) as the stage `medium-elo`: every stage of the course, the same suite, the same day.
- Play it in the browser: [https://rukh.borjaglez.com/?stage=medium-elo-fp16](https://rukh.borjaglez.com/?stage=medium-elo-fp16).
- Bring it to the paths the configs read: `uv run rukh pull medium-elo`.



## License

APACHE-2.0. The code and the weights are released under the Apache License 2.0;
the training data comes from Lichess under CC0. Please credit Lichess when you use them.

Generated with `rukh` 0.0.1.
