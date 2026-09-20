# `medium-v4-dpo-offpolicy` against `medium-v4-20260919-174623`

Two models playing each other over the same openings with the colours mirrored. This is
the instrument for a *difference* in strength: both models play the same game, so there
is no third party whose mood has to be averaged out.

| Metric | Value |
|---|---|
| Games | 400 |
| Score for `medium-v4-dpo-offpolicy` | 0.6012 |
| Won / drawn / lost | 199 / 83 / 118 |
| Elo difference | **+71** |
| 95 % CI | 41 to 102 |
| Separated from zero | yes |
| Illegal proposals | 511 by `medium-v4-dpo-offpolicy`, 268 by `medium-v4-20260919-174623` |

## By colour

A model that only wins with one colour has an opening repertoire, not an edge.

| `medium-v4-dpo-offpolicy` plays | Games | Score |
|---|---:|---:|
| white | 200 | 0.5825 |
| black | 200 | 0.6200 |

## Is this enough games?

An edge of 71 Elo takes **90** games to tell from nothing at 95 %, and this match played **400**. So the number above is a measurement.

## How it was played

Two numbers from different settings are two numbers, not a comparison.

| Setting | Value |
|---|---|
| Opening book | 6 plies, seed 13 |
| Elo header | `<w1800>` for both |
| Sampling | temperature 0.6, top-k 20 |
| Illegal moves | proposed unmasked, rescued with a masked draw |
| Bootstrap | 2000 resamples |

