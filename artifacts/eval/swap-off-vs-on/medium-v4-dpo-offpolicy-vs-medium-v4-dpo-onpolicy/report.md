# `medium-v4-dpo-offpolicy` against `medium-v4-dpo-onpolicy`

Two models playing each other over the same openings with the colours mirrored. This is
the instrument for a *difference* in strength: both models play the same game, so there
is no third party whose mood has to be averaged out.

| Metric | Value |
|---|---|
| Games | 400 |
| Score for `medium-v4-dpo-offpolicy` | 0.4625 |
| Won / drawn / lost | 142 / 86 / 172 |
| Elo difference | **-26** |
| 95 % CI | -57 to 3 |
| Separated from zero | no |
| Illegal proposals | 705 by `medium-v4-dpo-offpolicy`, 468 by `medium-v4-dpo-onpolicy` |

## By colour

A model that only wins with one colour has an opening repertoire, not an edge.

| `medium-v4-dpo-offpolicy` plays | Games | Score |
|---|---:|---:|
| white | 200 | 0.4800 |
| black | 200 | 0.4450 |

## Is this enough games?

An edge of 26 Elo takes **680** games to tell from nothing at 95 %, and this match played **400**. So the number above is a hint, not a measurement: play more games or say so.

