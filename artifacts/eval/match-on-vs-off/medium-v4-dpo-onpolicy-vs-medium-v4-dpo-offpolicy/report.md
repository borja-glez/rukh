# `medium-v4-dpo-onpolicy` against `medium-v4-dpo-offpolicy`

Two models playing each other over the same openings with the colours mirrored. This is
the instrument for a *difference* in strength: both models play the same game, so there
is no third party whose mood has to be averaged out.

| Metric | Value |
|---|---|
| Games | 400 |
| Score for `medium-v4-dpo-onpolicy` | 0.5687 |
| Won / drawn / lost | 191 / 73 / 136 |
| Elo difference | **+48** |
| 95 % CI | 19 to 80 |
| Separated from zero | yes |
| Illegal proposals | 441 by `medium-v4-dpo-onpolicy`, 692 by `medium-v4-dpo-offpolicy` |

## By colour

A model that only wins with one colour has an opening repertoire, not an edge.

| `medium-v4-dpo-onpolicy` plays | Games | Score |
|---|---:|---:|
| white | 200 | 0.5775 |
| black | 200 | 0.5600 |

## Is this enough games?

An edge of 48 Elo takes **200** games to tell from nothing at 95 %, and this match played **400**. So the number above is a measurement.

