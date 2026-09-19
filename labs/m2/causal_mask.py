"""Draw the causal mask and prove causality empirically on a real MoveDecoder."""

import sys

import torch

from rukh.models import DecoderConfig, MoveDecoder

# The Windows console is cp1252 by default and DuckDB draws its tables with box characters, so a
# plain `print` of a result set dies with UnicodeEncodeError. Ask for UTF-8 before printing
# anything; on a terminal that already speaks UTF-8 this is a no-op.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


T = 8
mask = torch.ones(T, T, dtype=torch.bool).tril()
print("Causal mask (row = query, column = key):")
print("      " + " ".join(f"{j:>2}" for j in range(T)))
for i in range(T):
    cells = " ".join(" #" if mask[i, j] else " ." for j in range(T))
    print(f"  q{i:<3} {cells}")
print(f"\nVisible pairs: {int(mask.sum())} of {T * T} ({100 * mask.float().mean():.1f} %)\n")

torch.manual_seed(0)
model = MoveDecoder(DecoderConfig(n_layer=2, n_head=2, d_model=32, vocab_size=64, block=T)).eval()
idx = torch.randint(1, 64, (1, T))

with torch.no_grad():
    base, _ = model(idx)
    for t in range(T - 1):
        changed = idx.clone()
        # Replace every token after t with a different one.
        changed[:, t + 1 :] = (changed[:, t + 1 :] + 7) % 63 + 1
        other, _ = model(changed)
        past = (base[:, : t + 1] - other[:, : t + 1]).abs().max().item()
        future = (base[:, t + 1 :] - other[:, t + 1 :]).abs().max().item()
        print(f"  cut after position {t}: max |delta| past {past:.3e}   future {future:.3e}")
        assert past == 0.0, "the past moved: the mask is not causal"
