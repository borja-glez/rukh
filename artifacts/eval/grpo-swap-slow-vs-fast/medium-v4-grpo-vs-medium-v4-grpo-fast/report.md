# `medium-v4-grpo` against `medium-v4-grpo-fast`

Two models playing each other over the same openings with the colours mirrored. This is
the instrument for a *difference* in strength: both models play the same game, so there
is no third party whose mood has to be averaged out.

| Metric | Value |
|---|---|
| Games | 400 |
| Score for `medium-v4-grpo` | 0.5125 |
| Won / drawn / lost | 163 / 84 / 153 |
| Elo difference | **+9** |
| 95 % CI | -22 to 40 |
| Separated from zero | no |
| Illegal proposals | 349 by `medium-v4-grpo`, 667 by `medium-v4-grpo-fast` |

## By colour

A model that only wins with one colour has an opening repertoire, not an edge.

| `medium-v4-grpo` plays | Games | Score |
|---|---:|---:|
| white | 200 | 0.4975 |
| black | 200 | 0.5275 |

## Is this enough games?

An edge of 9 Elo takes **6143** games to tell from nothing at 95 %, and this match played **400**. So the number above is a hint, not a measurement: play more games or say so.

## How it was played

Two numbers from different settings are two numbers, not a comparison.

| Setting | Value |
|---|---|
| Opening book | 6 plies, seed 7 |
| Elo header | `<w1800>` for both |
| Sampling | temperature 0.6, top-k 20 |
| Illegal moves | proposed unmasked, rescued with a masked draw |
| Bootstrap | 2000 resamples |

