# `medium-v4-dpo-offpolicy` against `medium-v4-20260919-174623`

Two models playing each other over the same openings with the colours mirrored. This is
the instrument for a *difference* in strength: both models play the same game, so there
is no third party whose mood has to be averaged out.

| Metric | Value |
|---|---|
| Games | 400 |
| Score for `medium-v4-dpo-offpolicy` | 0.5863 |
| Won / drawn / lost | 204 / 61 / 135 |
| Elo difference | **+61** |
| 95 % CI | 30 to 93 |
| Separated from zero | yes |
| Illegal proposals | 567 by `medium-v4-dpo-offpolicy`, 253 by `medium-v4-20260919-174623` |

## By colour

A model that only wins with one colour has an opening repertoire, not an edge.

| `medium-v4-dpo-offpolicy` plays | Games | Score |
|---|---:|---:|
| white | 200 | 0.6075 |
| black | 200 | 0.5650 |

## Is this enough games?

An edge of 61 Elo takes **126** games to tell from nothing at 95 %, and this match played **400**. So the number above is a measurement.

