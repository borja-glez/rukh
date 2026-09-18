import json
from pathlib import Path

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
