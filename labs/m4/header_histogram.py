"""Count how many times the model has seen each Elo header, before blaming the model.

Lab 1 of M4. The claim under test is "the Elo conditioning opens the axis but gives no strength",
which was the conclusion of the previous milestone. Before accepting it, count: the second token of
every packed game *is* White's Elo header, so a histogram of `tokens[starts + 1]` says exactly how
much training each condition got.

Run from the repository root. `--tokens` points at any packed stream:

    uv run python labs/m4/header_histogram.py
    uv run python labs/m4/header_histogram.py --tokens data/tokens-elo/uci/train

The corpus every published model trained on (`data/tokens-v4/uci/train`) starts at `<w1800>` and
has nothing below it: twelve header tokens that exist in the vocabulary, have a row in the
embedding table, and never received a gradient.
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

from rukh.tokenize.uci_vocab import UciTokenizer

# The Windows console is cp1252 by default; ask for UTF-8 before printing the bar chart.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tokens", default="data/tokens-v4/uci/train", type=Path)
parser.add_argument("--width", default=48, type=int, help="Columns of the widest bar.")
args = parser.parse_args()

tok = UciTokenizer()
tokens = np.load(args.tokens / "tokens.npy", mmap_mode="r")
starts = np.load(args.tokens / "starts.npy")

# Every game is `<bos> <wXXXX> <bXXXX> moves...`, so offset 1 from its start is White's header.
counts = collections.Counter(tokens[starts + 1].tolist())
if not counts:
    raise SystemExit(f"no games in {args.tokens}")

top = max(counts.values())
print(f"{args.tokens}: {len(starts):,} games, {len(tokens):,} tokens\n")
for token_id, n in sorted(counts.items()):
    bar = "#" * max(1, round(args.width * n / top))
    print(f"  {tok.ids[token_id]:>9s}  {n:>9,d}  {bar}")

# The twelve tokens the vocabulary has below 1800, and how many of them the corpus ever used.
below = [tok.vocab[f"<w{elo:04d}>"] for elo in range(600, 1800, 100)]
seen = sum(1 for token_id in below if counts.get(token_id))
print(f"\nheaders below <w1800>: {seen} of {len(below)} ever seen in this corpus")
print("an untrained header is not a header that does not work: it is noise with a row in a table")
