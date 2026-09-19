import json
import sys
from pathlib import Path

# The Windows console is cp1252 by default and DuckDB draws its tables with box characters, so a
# plain `print` of a result set dies with UnicodeEncodeError. Ask for UTF-8 before printing
# anything; on a terminal that already speaks UTF-8 this is a no-op.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


bpe = json.loads(Path("artifacts/tokenizer/bpe.json").read_text(encoding="utf-8"))
merges = bpe["model"]["merges"]
vocab = bpe["model"]["vocab"]

print("fusiones:", len(merges), " vocabulario:", len(vocab))
print("== primeras 10 fusiones (las más frecuentes) ==")
for pair in merges[:10]:
    print(" ", pair)

print("== 10 tokens más largos del vocabulario ==")
longest = sorted(vocab, key=len, reverse=True)[:10]
for token in longest:
    print(f"  {len(token):3d}  {token}")
