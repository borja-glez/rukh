# `medium-v4-grpo` against `medium-v4-20260919-174623`

Two models playing each other over the same openings with the colours mirrored. This is
the instrument for a *difference* in strength: both models play the same game, so there
is no third party whose mood has to be averaged out.

| Metric | Value |
|---|---|
| Games | 400 |
| Score for `medium-v4-grpo` | 0.5613 |
| Won / drawn / lost | 182 / 85 / 133 |
| Elo difference | **+43** |
| 95 % CI | 13 to 72 |
| Separated from zero | yes |
| Illegal proposals | 361 by `medium-v4-grpo`, 218 by `medium-v4-20260919-174623` |

## By colour

A model that only wins with one colour has an opening repertoire, not an edge.

| `medium-v4-grpo` plays | Games | Score |
|---|---:|---:|
| white | 200 | 0.6075 |
| black | 200 | 0.5150 |

## Is this enough games?

An edge of 43 Elo takes **253** games to tell from nothing at 95 %, and this match played **400**. So the number above is a measurement.

## How it was played

Two numbers from different settings are two numbers, not a comparison.

| Setting | Value |
|---|---|
| Opening book | 6 plies, seed 7 |
| Elo header | `<w1800>` for both |
| Sampling | temperature 0.6, top-k 20 |
| Illegal moves | proposed unmasked, rescued with a masked draw |
| Bootstrap | 2000 resamples |

