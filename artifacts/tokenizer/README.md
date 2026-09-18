---
license: cc0-1.0
language:
  - en
tags:
  - chess
  - tokenizer
---

# Rukh tokenizer

Tokenizers for chess games used by [Rukh](https://github.com/borja-glez/rukh), a chess language
model built from scratch. Three schemes encode the same game; the fixed UCI vocabulary is the
default and is implemented twice (Python in `rukh`, TypeScript in `rukh-web`/`rukh-lab`), so the
fixture in this folder is the parity test between both.

## Files

| File | What |
|---|---|
| `vocab.json` | Fixed UCI vocabulary: `{"version": 1, "size": 2030, "specials": [...], "elo_bins": {"min": 600, "max": 3200, "step": 100}, "tokens": [... 2030 strings in id order ...]}` |
| `bpe.json` | Hugging Face `tokenizers` BPE trained on UCI move text (`model.vocab`, `model.merges`) |
| `fixtures/games.json` | 20 games encoded with every scheme (see below) |

## Scheme 1: fixed UCI vocabulary (`vocab.json`)

The vocabulary is an enumeration, not something learned from data:

1. Squares in file-major order: `a1, a2, ..., a8, b1, ..., h8` (index `file * 8 + rank`).
2. Moves: for each `from` in that order, for each `to` in that order with `to != from`, the string
   `from + to` is included when `to` is reachable from `from` by a queen (same file, rank or
   diagonal) or by a knight. That gives 1 792 strings.
3. Promotions: for each `from` on rank 7 (White) and then rank 2 (Black), in square order; for
   each `to` on rank 8 (resp. 1) with `|Δfile| <= 1`, in square order; for each piece in
   `q, r, b, n`: `from + to + piece`. That gives 176 strings. Moves total 1 968.
4. Special tokens come first: `<pad>`=0, `<bos>`=1, `<eos>`=2, `<mask>`=3, `<unk>`=4, `<1-0>`=5,
   `<0-1>`=6, `<1/2>`=7, then `<w0600>` ... `<w3200>` (27 bins of 100 Elo, ids 8-34) and
   `<b0600>` ... `<b3200>` (ids 35-61). Moves start at id 62. Total size 2 030.
5. Elo to bin: `min(max(elo, 600), 3299) // 100 * 100`, formatted with four digits.

A game is encoded as `[<bos>, <wXXXX>, <bXXXX>, move..., <result>, <eos>]`, cut to `max_len`
(200 by default): when the whole game fits it ends with `<eos>`, otherwise it is truncated to the
first `max_len` ids. Moves not in the vocabulary map to `<unk>`.

## Scheme 2: char-level SAN

Fixed alphabet of 32 characters, `" #+-.012345678=BKNOQRabcdefghx/"`, plus `<pad>`=0, `<bos>`=1,
`<eos>`=2 in front (ids 3-34 are the characters, in the order above). The text is the numbered
SAN movetext without comments followed by a space and the result: `1.e4 e5 2.Nf3 ... 1-0`.
`encode` wraps the characters in `<bos>` ... `<eos>`.

## Scheme 3: BPE over UCI text (`bpe.json`)

Trained with `tokenizers`: `models.BPE(unk_token="<unk>")`, `pre_tokenizers.WhitespaceSplit()`,
`trainers.BpeTrainer(vocab_size, special_tokens=<the 8 specials above>)`, no normalizer, no
continuing-subword prefix, no end-of-word suffix. The text is the plain UCI move string
(`e2e4 e7e5 ...`); merges apply by rank inside each whitespace-split word, so multi-move tokens
such as `e2e4e7e5` never appear (a word is a single move), but frequent moves become one token.
Ids are those of `Tokenizer.encode(text).ids`, with no `<bos>`/`<eos>` added.

## Fixture format (`fixtures/games.json`)

A JSON list with one object per game:

```json
{
  "id": "opera-1858",
  "white_elo": 2600,
  "black_elo": 2000,
  "result": "1-0",
  "uci": "e2e4 e7e5 g1f3 ...",
  "san": "1.e4 e5 2.Nf3 ...",
  "uci_ids": [1, 28, 49, 997, ...],
  "san_ids": [1, 4, 10, ...],
  "bpe_ids": [...]
}
```

- `uci_ids = UciTokenizer().encode_game(uci, white_elo, black_elo, result, max_len=200)`
- `san_ids = SanCharTokenizer().encode(san + " " + result)`
- `bpe_ids = Tokenizer.from_file("bpe.json").encode(uci).ids`

The games come from `tests/fixtures/games.pgn` in the `rukh` repository and cover castling on
both wings, promotions (including an underpromotion to a knight), en passant, captures with
check, a 4-ply game, a game longer than 200 plies (so the `max_len` cut is exercised), and the
three results.
