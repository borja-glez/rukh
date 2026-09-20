---
license: apache-2.0
library_name: rukh
base_model: chorcat/rukh-medium-lora
tags:
  - chess
  - rukh
  - lora
  - adapter
---

# chorcat/rukh-lora-e4

A LoRA adapter for [`chorcat/rukh-medium-lora`](https://huggingface.co/chorcat/rukh-medium-lora). On its own it does
nothing: it is 393,216 numbers (1.6 MB) that
correct that model's weights, and without them there is no model to correct.

Written by hand rather than imported, as part of [Rukh](https://github.com/borja-glez/rukh), a course that builds
a chess language model end to end. How it works and why: [https://lab.rukh.borjaglez.com](https://lab.rukh.borjaglez.com).

## What it is

| | |
|---|---|
| Rank `r` | 8 |
| Scaling `alpha` | 16 (16/8 applied to `B·A`) |
| Adapted matrices | `q, v`, in every block |
| Trainable parameters | 393,216 |
| File size | 1.6 MB |

The base weights never move. Everything outside `A` and `B` is frozen during training, which is
what makes the file this small and what limits what it can learn: the correction has to pass
through 8 dimensions per matrix.

## What it changed

The model's own probability for `e2e4` from the opening position, read off the
softmax over the twenty legal first moves with no sampling -- so the number is a property of the
weights and two readings agree to the last decimal:

| | P(`e2e4`) | first-move entropy |
|---|---:|---:|
| `chorcat/rukh-medium-lora` | 59.64 % | 1.7695 bits |
| with this adapter | 99.85 % | 0.0209 bits |

And what the style cost, on the same suite every other stage of the project is measured with --
same validation positions, same puzzles, same seed:

| | `chorcat/rukh-medium-lora` | with this adapter |
|---|---:|---:|
| Legality without the mask, argmax | 99.8 % | 99.8 % |
| Top-1 next move | 54.4 % | 54.7 % |
| Puzzles solved | 37.5 % | 37.8 % |


## How it was trained

200,000 games selected with `split_part(uci, ' ', 1) = 'e2e4' AND white_elo >= 1800 AND black_elo >= 1800`, from the
1800+ Lichess corpus of the project.

## Using it

```python
from rukh.models import MoveDecoder
from rukh.models.lora import load_adapter
from rukh.train import load_model

model, _ = load_model("<the base checkpoint>")
load_adapter(model, "adapter.safetensors")  # the config travels next to it
```

`rukh.models.lora.merge_lora(model)` folds it into the weights, after which the model is an
ordinary `MoveDecoder`: same module names, same state dict, exportable to ONNX like any other.

### In the browser

`web/adapter.bin` is the same factors again as one flat little-endian `float32` buffer
(1.6 MB): `A` first, then `B`, both stacked over the blocks in the
order they run, with the `alpha/r` scaling already applied. `web/adapter.json` carries the shapes,
because a buffer of floats says nothing about itself.

It is meant for the ONNX export of `chorcat/rukh-medium-lora` that takes its LoRA factors as **inputs of the
graph** (`lora_a` and `lora_b`) rather than baked into the weights. Fed an adapter of zeros that
file is the base model exactly; fed this one it is this style. So the demo changes style by
downloading 1.6 MB instead of a second copy of the model, which is
the only reason a low-rank correction is worth keeping low-rank once it leaves the training loop.

## Licence

apache-2.0, the same as the base model.
