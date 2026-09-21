---
license: apache-2.0
library_name: rukh
pipeline_tag: feature-extraction
language:
  - en
datasets:
  - chorcat/rukh-games-1800
  - chorcat/rukh-tokenizer
tags:
  - chess
  - rukh
  - encoder
  - onnx
---

# chorcat/rukh-encoder

A bidirectional transformer written from scratch that reads a chess **position** and says three
things about it: how good it is, whether the move that led to it threw the game away, and how
the game is going to end. This is the `encoder-v4` stage of [Rukh](https://github.com/borja-glez/rukh), a
course that builds a chess language model end to end: 38,973,957 parameters
over 12 layers of width 512,
pretrained with masked move modeling and fine-tuned on Stockfish
labels.

See it evaluate a live game: [https://rukh.borjaglez.com](https://rukh.borjaglez.com) · read how it was built:
[https://lab.rukh.borjaglez.com](https://lab.rukh.borjaglez.com)

## Results

Measured with `rukh eval encoder` on 2026-09-21, over
10000 held-out positions (7453 of them with a blunder label).

| Metric | Value |
|---|---|
| Blunder F1 (tuned, `p >= 0.09617`) | 18.6 % |
| Blunder precision | 13.8 % |
| Blunder recall | 28.5 % |
| Blunder F1 (fixed, `p >= 0.5`) | 0.0 % |
| Blunder ROC AUC | 0.761 |
| Blunder average precision | 0.117 |
| Blunder base rate | 3.7 % |
| Blunder F1, material baseline | 8.9 % |
| Margin over the baseline | +9.7 F1 points |
| Value vs Stockfish cp, Pearson | 0.726 |
| Value vs Stockfish cp, Spearman | 0.666 |
| Result accuracy | 50.8 % |

### Acceptance bars

The project set these two bars for the encoder in `GOAL.md` before any of this was trained. This
is where the release stands against them, and it is the same table whether the answer is
flattering or not.

| Bar | Target | Measured | Verdict |
|---|---|---|---|
| Blunder F1 over the material baseline | at least +5 F1 points | +9.7 F1 points | met |
| Value vs Stockfish cp, Spearman | at least 0.80 | 0.666 | **not met** |

**The value bar is not met.** The head was fine-tuned for 4 000 steps on labels that cover 9.8 %
of the positions they were crossed with, and the target `tanh(cp / 400)` compresses the scale
exactly where the positions are densest. It is the same diagnosis as the decoder's: what is
missing is labelled data, not capacity.

### How the blunder numbers were measured

A blunder is rare (3.7 % of the labelled rows), so **accuracy is meaningless** here: a model that always answers "no blunder" scores 96.3 % without knowing anything about chess. F1 at an arbitrary threshold is nearly as bad, because an uncalibrated sigmoid can rank the positions well and still put every probability below 0.5: that number measures the operating point, not the representation. This is why the threshold is chosen on a `tune` half and the F1 is reported on a `score` half, and why ROC AUC and average precision, which no threshold can flatter, are reported next to it.

The labelled rows are cut in two **by game**, never by position: the `tune` half
(3793 rows, 2455 games) chose the threshold `p >= 0.09617140889167786`
by maximising F1 there, and the `score` half (3660 rows, 2368 games) is
where the F1, the precision and the recall in the table are measured. No game is in both halves,
and the threshold never saw the rows it is scored on. The F1 at the fixed threshold is in the
same table so that the operating point cannot hide anything.

The material baseline is a hard yes/no rule: it has no threshold, so nothing was tuned on its
side and it was given no half to tune on. The margin compares the model at its best operating
point against the rule at its only one.

### The baseline these numbers are compared against

The blunder head is only interesting if it beats a program that understands nothing. The
baseline in the table is deliberately the dumbest thing that can judge a move: **material**
(pawn 1, knight 3, bishop 3, rook 5, queen 9) plus **mobility** (legal moves, 0.05 of a pawn
each), calling a move a blunder when it loses at least one point of net material after one ply
of the opponent's captures. It is measured on exactly the same rows as the model, and the
project's target is five F1 points above it.

What the baseline cannot see is the point: a positional sacrifice is invisible to it. Given
Fischer's 17...Be6 against Byrne (New York, 1956), it sees a queen hanging and calls one of the
most famous moves in chess a nine-point blunder.

The held-out split is drawn **by game**, never by position: two positions of the same game are
the same game one move apart, and splitting between them would leak.


How to read these numbers:

- the blunder F1 of the encoder and of the material baseline are measured on the same rows of the held-out 'val' split, which is drawn by game_id, never by position, and those rows are cut in two by game_id again: the 'tune' half chooses the threshold and the 'score' half is what gets reported
- the checkpoint carries no label-count curve: run `rukh train heads --curve` and evaluate one of its checkpoints to fill that table
- the encoder is +9.7 F1 points from the baseline; GOAL.md asks for at least +5
- the value head is correlated against `tanh(cp / 400)`, the bounded score it is trained on, and not against raw `cp`, where a forced mate is worth ±9 99x and a few rows would decide Pearson for the whole set; the GOAL.md bar of 0.80 is read on Spearman
- the baseline counts material (1/3/3/5/9) and mobility and looks one ply ahead at captures: it cannot see a positional sacrifice, and calls Fischer's 17...Be6 (Byrne-Fischer, 1956) a nine-point blunder
- the two blunder detectors do not see the same thing: the baseline is given the predecessor position and the move that was played, while the encoder is given only the resulting position and has to infer that something was thrown away. That is the comparison GOAL.md asks for, but it is not a level playing field
- the blunder threshold 0.09617 was chosen on the 'tune' half (3,793 rows, 2,455 games) by maximising F1 there, and the reported numbers are measured on the 'score' half (3,660 rows, 2,368 games), which no threshold ever saw; the same rows at the fixed threshold 0.5 give an F1 of 0.0000 against 0.1855 tuned
- the material baseline is a hard yes/no rule: it has no threshold, so nothing was tuned on its side and it got no half to tune on. The margin therefore compares a model at its best operating point against a rule at its only one
- a blunder is rare (3.7 % of the labelled rows), so **accuracy is meaningless** here: a model that always answers "no blunder" scores 96.3 % without knowing anything about chess. F1 at an arbitrary threshold is nearly as bad, because an uncalibrated sigmoid can rank the positions well and still put every probability below 0.5: that number measures the operating point, not the representation. This is why the threshold is chosen on a `tune` half and the F1 is reported on a `score` half, and why ROC AUC and average precision, which no threshold can flatter, are reported next to it


## Input and output

The model reads the `moves` scheme.

A position is the game that led to it, in the same UCI vocabulary the decoder was trained on,
which ships in `tokenizer/vocab.json`: `<bos>`, the two Elo tokens, and the moves up to the
position, cropped from the left while keeping those three header tokens.

One caveat that comes with this scheme: the supervised table is deduplicated by four-field FEN,
so the prefix is **a** line that reaches the position, not necessarily the one the labelled game
played. The position, the value and the blunder verdict are the same either way; the history may
not be.

Tokens are pooled with `mean` pooling into one vector per position, and three linear
heads read that vector: `value`, `blunder`, `result`.
`value` is `tanh(cp / 400)` from White's point of view, `blunder` is a logit in PyTorch (and a
probability in the exported graph), and `result` is three classes (White, draw, Black).

The pooled vector is itself a released output: it is the position embedding used to retrieve
similar positions (`rukh encoder embed --positions <parquet> --out <npy>`).

## Files

- `model.safetensors`
- `config.json`
- `tokenizer/vocab.json`
- `onnx/model-fp16.onnx`
- `onnx/model-int8.onnx`
- `onnx/model.onnx`
- `onnx/parity.json`

The ONNX graph returns **two** outputs, `value` and `blunder`, because that is all the demo's
evaluation bar and blunder alert need; `blunder` comes out as a probability, so the page
compares it against 0.5. `model-fp16.onnx` is for WebGPU and `model-int8.onnx` for the WASM
fallback. The metadata carries `rukh_kind=encoder` and `rukh_heads`.

### How faithful the ONNX files are

Every exported file was run against the PyTorch checkpoint on 1000
held-out labelled positions, comparing the blunder decision at p >= 0.5. `onnx/parity.json` in this
repository is that measurement, as the exporter wrote it.

| File | Same blunder decision as PyTorch | Worst `value` drift |
|---|---|---|
| `model.onnx` (fp32) | 100.0 % | 4.44e-06 |
| `model-fp16.onnx` (fp16) | 100.0 % | 0.00164 |
| `model-int8.onnx` (int8) | 100.0 % | 0.117 |

The bar the project set itself is 99.9 %.
Every precision, int8 included, makes the same call on every position checked: quantizing this
model costs nothing that the demo's blunder alert can see. That is worth stating next to the
decoder's number, where it is not true.

## Training recipe


| Parameter | Value |
|---|---|
| `batch_size` | `128` |
| `betas` | `[0.9, 0.95]` |
| `block` | `200` |
| `ckpt_every` | `1000` |
| `compile` | `False` |
| `curve` | `[]` |
| `device` | `cuda` |
| `encoder_ckpt` | `checkpoints/encoder-mmm-v4-20260920-111628/best.pt` |
| `eval_batches` | `50` |
| `eval_every` | `250` |
| `fraction` | `1.0` |
| `grad_accum` | `1` |
| `grad_clip` | `1.0` |
| `input` | `moves` |
| `labels.blunder_cp` | `100` |
| `labels.games_dir` | `data/uci` |
| `labels.out_dir` | `data/labels` |
| `labels.positions_eval` | `data/evals/positions-eval.parquet` |
| `labels.seed` | `42` |
| `labels.val_fraction` | `0.1` |
| `labels.value_scale` | `400.0` |
| `last_n` | `2` |
| `log_every` | `10` |
| `lr` | `0.001` |
| `max_steps` | `4000` |
| `min_lr_ratio` | `0.1` |
| `mode` | `last-n` |
| `model` | `None` |
| `num_params` | `38973957` |
| `out_dir` | `checkpoints` |
| `pooling` | `mean` |
| `precision` | `bf16` |
| `run_name` | `encoder-heads-v4` |
| `seed` | `42` |
| `train_labels` | `438093` |
| `trainable_encoder_tensors` | `26` |
| `unique_run_name` | `True` |
| `val_labels` | `50066` |
| `warmup` | `200` |
| `weight_decay` | `0.01` |
| `weights.blunder` | `1.0` |
| `weights.result` | `0.5` |
| `weights.value` | `1.0` |
| `weights.value_rank` | `1.0` |
| `workers` | `4` |

MLflow run: `647cdcf0d6b1411b96190312561aaf62`.

```json
{
  "architectures": [
    "PositionEncoder"
  ],
  "model_type": "rukh-position-encoder",
  "library_name": "rukh",
  "rukh_version": "0.0.1",
  "stage": "encoder-v4",
  "step": 4000,
  "params": 38973957,
  "tokenizer": "moves",
  "vocab_hash": null,
  "data_manifest_sha": null,
  "git_sha": "db5d48e5db36a0d24e00cbe21375167b8f9f9c3b",
  "heads": [
    "value",
    "blunder",
    "result"
  ],
  "pooling": "mean",
  "pretrained_from": "checkpoints/encoder-mmm-v4-20260920-111628/best.pt",
  "input": "moves",
  "vocab_size": 2030,
  "square_vocab": 47,
  "n_layer": 12,
  "n_head": 8,
  "d_model": 512,
  "d_ff": null,
  "block": 200,
  "dropout": 0.1,
  "pos": "learned",
  "tie_embeddings": true
}
```

## Data

Pretrained with masked move modeling on [`chorcat/rukh-games-1800`](https://huggingface.co/datasets/chorcat/rukh-games-1800), [`chorcat/rukh-tokenizer`](https://huggingface.co/datasets/chorcat/rukh-tokenizer),
derived from the [Lichess open database](https://database.lichess.org) (CC0), and fine-tuned on
positions
crossed with the Lichess Stockfish evaluations: `value` and `blunder` come from those
scores, `result` from the game the position was played in.

The pretraining checkpoint the heads started from: `checkpoints/encoder-mmm-v4-20260920-111628/best.pt`.


## Limitations

- It has no search. It judges a position from the position, so a tactic that needs three moves
  to appear is one it can only guess at.
- The blunder label is a threshold on a Stockfish score (100 centipawns lost against the best
  line of the previous position), not a human judgement of what counts as a mistake.
- `result` is the noisiest of the three heads: every position of a game carries the same label,
  including the ones played before anything was decided.
- The evaluations come from community analysis of varying depth, so the `value` head inherits
  whatever bias that has.
- It was trained on games between 1800+ humans on Lichess; the positions it knows best are the
  positions those games reach.

## In the course

- Built in [M3 · El encoder: los labs](https://lab.rukh.borjaglez.com/curso/m3/03-labs-del-encoder/), the lesson that runs every command behind this repository.
- Measured in the [single results table](https://lab.rukh.borjaglez.com/proyecto/) as the stage `encoder-v4`: every stage of the course, the same suite, the same day.
- Bring it to the paths the configs read: `uv run rukh pull encoder-v4`.



## License

APACHE-2.0. The code and the weights are released under the Apache License 2.0;
the training data comes from Lichess under CC0. Please credit Lichess when you use them.

Generated with `rukh` 0.0.1.
