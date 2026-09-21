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
  - masked-move-modeling
---

# chorcat/rukh-encoder-mmm

A bidirectional transformer written from scratch and **pretrained only**: it has no value,
blunder or result head yet. It read games written in UCI with 15 % of the moves hidden and
learned to put them back by looking at both sides of the hole, which is masked language
modeling with moves for words. This is the `encoder-mmm-v4` stage of [Rukh](https://github.com/borja-glez/rukh),
a course that builds a chess language model end to end: 38,971,392 parameters
over 12 layers of width 512, on the
`moves` input scheme.

It exists on the Hub for one reason: it is where the encoder with heads
([`chorcat/rukh-encoder`](https://huggingface.co/chorcat/rukh-encoder)) starts from, and a reader of the
course who skips the pretraining lab can put it where the fine-tuning configs look for it:

```sh
uv run rukh pull encoder-mmm-v4      # -> checkpoints/encoder-mmm-v4/best.pt
uv run rukh train heads --config configs/train/encoder-heads-v4.yaml --mode last-n
```

Read how it was built: [https://lab.rukh.borjaglez.com](https://lab.rukh.borjaglez.com)

## Results

The only thing a pretraining run measures is how well it fills the holes it was given.

| Metric | Value |
|---|---|
| Masked-move loss on held-out games | 0.6981 |
| Masked-move top-1 on held-out games | 81.4 % |
| Training step of this checkpoint | 16000 |

These are not the numbers of the model that goes into the demo: the heads are fine-tuned on
Stockfish labels afterwards, and their F1, correlation and result accuracy live on the card of
[`chorcat/rukh-encoder`](https://huggingface.co/chorcat/rukh-encoder).

## Training recipe


| Parameter | Value |
|---|---|
| `batch_size` | `96` |
| `betas` | `[0.9, 0.95]` |
| `block` | `200` |
| `ckpt_every` | `1000` |
| `compile` | `True` |
| `data_manifest_sha` | `aa0809618dbc372d3cc192fef9dd110dca1ce4aae22dca13ea648c02d78c10b4` |
| `device` | `cuda` |
| `eval_batches` | `50` |
| `eval_every` | `500` |
| `grad_accum` | `3` |
| `grad_clip` | `1.0` |
| `input` | `moves` |
| `log_every` | `10` |
| `lr` | `0.0005` |
| `masking.keep_ratio` | `0.1` |
| `masking.mask_ratio` | `0.8` |
| `masking.prob` | `0.15` |
| `masking.random_ratio` | `0.1` |
| `masking.seed` | `0` |
| `max_steps` | `16000` |
| `min_lr_ratio` | `0.1` |
| `model.block` | `200` |
| `model.d_ff` | `None` |
| `model.d_model` | `512` |
| `model.dropout` | `0.1` |
| `model.input` | `moves` |
| `model.n_head` | `8` |
| `model.n_layer` | `12` |
| `model.pos` | `learned` |
| `model.square_vocab` | `47` |
| `model.tie_embeddings` | `True` |
| `model.vocab_size` | `2030` |
| `num_params` | `38868992` |
| `out_dir` | `checkpoints` |
| `precision` | `bf16` |
| `run_name` | `encoder-mmm-v4` |
| `seed` | `42` |
| `tokens_dir` | `data/tokens-v4/uci` |
| `unique_run_name` | `True` |
| `vocab_hash` | `527c5dda224cab570cb84da8f7dcda0f43fe53edaf86822f899568f5dd824b46` |
| `warmup` | `1000` |
| `weight_decay` | `0.1` |
| `workers` | `4` |

MLflow run: `89ccdb9ab6264c8c8301f5d6392304e2`.

```json
{
  "architectures": [
    "PositionEncoder"
  ],
  "model_type": "rukh-position-encoder",
  "library_name": "rukh",
  "rukh_version": "0.0.1",
  "stage": "encoder-mmm-v4",
  "step": 16000,
  "params": 38971392,
  "tokenizer": "moves",
  "vocab_hash": "527c5dda224cab570cb84da8f7dcda0f43fe53edaf86822f899568f5dd824b46",
  "data_manifest_sha": "aa0809618dbc372d3cc192fef9dd110dca1ce4aae22dca13ea648c02d78c10b4",
  "git_sha": "db5d48e5db36a0d24e00cbe21375167b8f9f9c3b",
  "heads": [
    "value",
    "blunder",
    "result"
  ],
  "pooling": "mean",
  "pretrained_from": null,
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

Trained on [`chorcat/rukh-games-1800`](https://huggingface.co/datasets/chorcat/rukh-games-1800), [`chorcat/rukh-tokenizer`](https://huggingface.co/datasets/chorcat/rukh-tokenizer),
derived from the [Lichess open database](https://database.lichess.org) (CC0), on the same
packed token stream the project's decoder trains on, so the two models saw the same games.
Validation uses a held-out slice of a month the training split does not read.

## Limitations

- It predicts hidden moves; it does not evaluate positions. Ask it how good a position is and
  the answer is whatever an untrained linear head returns.
- Its masked-move accuracy is not a measure of chess strength: a move is easy to fill in when
  the moves around it leave one legal option.
- Fine-tuning the whole network on the labels afterwards made it *worse* than fine-tuning the
  last two blocks; the course records that measurement, and the heads configs default to it.

## License

Weights: Apache-2.0. Data: CC0 (Lichess). Code: MIT, at [https://github.com/borja-glez/rukh](https://github.com/borja-glez/rukh).
