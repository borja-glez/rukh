---
license: apache-2.0
library_name: rukh
pipeline_tag: feature-extraction
language:
  - en
datasets:
  - chorcat/rukh-pairs-dpo
tags:
  - chess
  - rukh
  - reward-model
  - preference
---

# chorcat/rukh-rm

A **reward model**: it reads a chess position and answers with one number, trained on
13,838 pairs of moves that a chess engine had already ranked. This is the
`rm` stage of [Rukh](https://github.com/borja-glez/rukh), a course that builds a chess language model
end to end: 5,937 parameters over 1 layers of width
16, reading the 69 square tokens of a board.

Read how it was built: [https://lab.rukh.borjaglez.com](https://lab.rukh.borjaglez.com)

## What the number means, and what it does not

The training objective is Bradley-Terry: `-log sigma(r(chosen) - r(rejected))`. Only
**differences** appear in it, so adding a constant to every score changes nothing. The scale has
no units and no origin, and comparing this model's `0.8` with another training run's `1.4` is
comparing two rulers with no zero. What the model is for is **ordering two positions**, and the
number below is how often it gets that order right.

## Results

Measured on 1,462 held-out pairs, split **by game** so that two pairs a
few plies apart cannot straddle the split.

| Metric | Value |
|---|---|
| Accuracy on held-out pairs | 72.91 % |
| Accuracy over pairs an evaluation can decide | 73.62 % |
| Preference loss | 0.5244 |
| Margin vs engine gap (Pearson) | -0.034 |
| Margin vs engine gap, without mates (Pearson) | +0.058 |

### Where it is right and where it is wrong

Accuracy by how far apart the engine scored the two moves. The **mate** band is the one to read
first: it is the largest slice after the narrowest band, and it is where the model is worst.

| Engine gap | Pairs | Accuracy |
|---|---:|---:|
| 100-200 cp | 616 | 74.19 % |
| mate | 484 | 71.49 % |

Read the `Pairs` column before the `Accuracy` one. A band of a dozen pairs reports whatever those
dozen did: across five runs of this configuration the `800-2000` band gave 90 %, 68 %, 90 %, 50 %
and 57 %, which is the same absence of data wearing five faces.

A mate is a tactical fact, and what this model reads is a static view of the position the move
leads to: it cannot see one. That band is exactly what a **verifiable** reward answers without
error, which is why the same milestone that trained this model also wrote one.

The two correlations above do not share a sign, and that is not a typo. The mate pairs carry the
largest engine gap by construction and the smallest learned margin, so a third of the points bends
the line the other way. One correlation over two regimes describes neither.

## What it is not for

**DPO does not need this model.** Direct Preference Optimization has an implicit reference -- a
frozen copy of the starting policy -- and trains on the pairs directly. A reward model is what PPO
needs, what a learned reward is compared against, and what makes the accuracy above a number worth
publishing at all.

## Training

13,838 pairs, 12,376 for training and
1,462 held out, 10 epochs, batch 64, learning rate
0.0003, seed 42.

Trained **from a random initialisation**, and that was measured rather than
assumed: starting from the project's pretrained position encoder scored about 6.5 points *worse*.
That encoder learned to fill in masked moves; this task is ordering positions by how good they
are, and what it brought did not help.

The seed also decides the split, so it decides how many mate pairs land in validation -- and the
headline number moves with that. Four seeds of this exact configuration gave
72.91 %, 74.21 %, 74.71 %, 74.76 % and 75.19 %. The first two are the **same seed**, so 1.3 of those points are not the split: they are what this measurement repeats to. Compare two runs only if they shared a seed.

## Reproducing

```bash
uv run rukh train reward --config configs/train/rm.yaml
```

## Licence

apache-2.0. Trained on engine evaluations of public Lichess games.
