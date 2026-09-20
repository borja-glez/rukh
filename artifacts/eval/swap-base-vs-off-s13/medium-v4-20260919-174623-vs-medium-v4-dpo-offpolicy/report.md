# `medium-v4-20260919-174623` against `medium-v4-dpo-offpolicy`

Two models playing each other over the same openings with the colours mirrored. This is
the instrument for a *difference* in strength: both models play the same game, so there
is no third party whose mood has to be averaged out.

| Metric | Value |
|---|---|
| Games | 400 |
| Score for `medium-v4-20260919-174623` | 0.4213 |
| Won / drawn / lost | 127 / 83 / 190 |
| Elo difference | **-55** |
| 95 % CI | -86 to -25 |
| Separated from zero | yes |
| Illegal proposals | 219 by `medium-v4-20260919-174623`, 490 by `medium-v4-dpo-offpolicy` |

## By colour

A model that only wins with one colour has an opening repertoire, not an edge.

| `medium-v4-20260919-174623` plays | Games | Score |
|---|---:|---:|
| white | 200 | 0.4525 |
| black | 200 | 0.3900 |

## Is this enough games?

An edge of 55 Elo takes **152** games to tell from nothing at 95 %, and this match played **400**. So the number above is a measurement.

## How it was played

Two numbers from different settings are two numbers, not a comparison.

| Setting | Value |
|---|---|
| Opening book | 6 plies, seed 13 |
| Elo header | `<w1800>` for both |
| Sampling | temperature 0.6, top-k 20 |
| Illegal moves | proposed unmasked, rescued with a masked draw |
| Bootstrap | 2000 resamples |

