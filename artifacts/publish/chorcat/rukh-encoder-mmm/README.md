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
modeling with moves for words. This is the `encoder-mmm` stage of [Rukh](https://github.com/borja-glez/rukh),
a course that builds a chess language model end to end: 38,971,392 parameters
over 12 layers of width 512, on the
`moves` input scheme.

It exists on the Hub for one reason: it is where the encoder with heads
([`chorcat/rukh-encoder`](https://huggingface.co/chorcat/rukh-encoder)) starts from, and a reader of the
course who skips the pretraining lab can put it where the fine-tuning configs look for it:

```sh
uv run rukh pull encoder-mmm      # -> checkpoints/encoder-mmm/best.pt
uv run rukh train heads --config configs/train/encoder-heads-v4.yaml --mode last-n
```

Read how it was built: [https://lab.rukh.borjaglez.com](https://lab.rukh.borjaglez.com)

## Results

The only thing a pretraining run measures is how well it fills the holes it was given.

| Metric | Value |
|---|---|
| Masked-move loss on held-out games | n/a |
| Masked-move top-1 on held-out games | n/a |
| Training step of this checkpoint | 16000 |

These are not the numbers of the model that goes into the demo: the heads are fine-tuned on
Stockfish labels afterwards, and their F1, correlation and result accuracy live on the card of
[`chorcat/rukh-encoder`](https://huggingface.co/chorcat/rukh-encoder).

## Training recipe


The MLflow run for this checkpoint was not available when the card was generated; the shape of
the model is in `config.json`.



```json
{
  "architectures": [
    "PositionEncoder"
  ],
  "model_type": "rukh-position-encoder",
  "library_name": "rukh",
  "rukh_version": "0.0.1",
  "stage": "encoder-mmm",
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

## In the course

- Built in [M3 · El encoder: los labs](https://lab.rukh.borjaglez.com/curso/m3/03-labs-del-encoder/), the lesson that runs every command behind this repository.
- Measured in the [single results table](https://lab.rukh.borjaglez.com/proyecto/): every stage of the course, the same suite, the same day.
- Bring it to the paths the configs read: `uv run rukh pull encoder-mmm-v4`.



## License

Weights: Apache-2.0. Data: CC0 (Lichess). Code: MIT, at [https://github.com/borja-glez/rukh](https://github.com/borja-glez/rukh).
