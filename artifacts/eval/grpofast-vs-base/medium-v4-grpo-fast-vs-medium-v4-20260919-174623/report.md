# `medium-v4-grpo-fast` against `medium-v4-20260919-174623`

Two models playing each other over the same openings with the colours mirrored. This is
the instrument for a *difference* in strength: both models play the same game, so there
is no third party whose mood has to be averaged out.

| Metric | Value |
|---|---|
| Games | 400 |
| Score for `medium-v4-grpo-fast` | 0.5775 |
| Won / drawn / lost | 190 / 82 / 128 |
| Elo difference | **+54** |
| 95 % CI | 25 to 85 |
| Separated from zero | yes |
| Illegal proposals | 572 by `medium-v4-grpo-fast`, 275 by `medium-v4-20260919-174623` |

## By colour

A model that only wins with one colour has an opening repertoire, not an edge.

| `medium-v4-grpo-fast` plays | Games | Score |
|---|---:|---:|
| white | 200 | 0.6200 |
| black | 200 | 0.5350 |

## Is this enough games?

An edge of 54 Elo takes **157** games to tell from nothing at 95 %, and this match played **400**. So the number above is a measurement.

## How it was played

Two numbers from different settings are two numbers, not a comparison.

| Setting | Value |
|---|---|
| Opening book | 6 plies, seed 7 |
| Elo header | `<w1800>` for both |
| Sampling | temperature 0.6, top-k 20 |
| Illegal moves | proposed unmasked, rescued with a masked draw |
| Bootstrap | 2000 resamples |

