"""Show that the encoder is not causal, and that the masking recipe does what it claims."""

import sys

import torch

from rukh.models import EncoderConfig, PositionEncoder
from rukh.models.encoder import MMM_IGNORE_INDEX
from rukh.train.mmm import MaskingConfig, apply_masking, control_ids, mask_id, masking_generator

# The Windows console is cp1252 by default and DuckDB draws its tables with box characters, so a
# plain `print` of a result set dies with UnicodeEncodeError. Ask for UTF-8 before printing
# anything; on a terminal that already speaks UTF-8 this is a no-op.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


torch.manual_seed(0)
cfg = EncoderConfig(n_layer=2, n_head=2, d_model=64, block=16)
model = PositionEncoder(cfg).eval()
print(f"preset parameters: {PositionEncoder(EncoderConfig()).num_params(False):,}")

# 1. Bidirectionality: change the LAST token and watch the FIRST hidden state move.
idx = torch.randint(100, cfg.vocab_size, (1, 8))
with torch.no_grad():
    before = model(idx)[0, 0].clone()
    idx[0, -1] = (idx[0, -1] + 1) % cfg.vocab_size
    after = model(idx)[0, 0]
delta = (before - after).abs().max().item()
print(f"max |delta| on the FIRST token after changing the LAST one: {delta:.6f}")
print("causal" if delta == 0.0 else "bidirectional: the future reaches the past")

# 2. Padding: the real tokens of a sequence must not depend on how much padding travels with it.
short = torch.tensor([[7, 8, 9, 0, 0, 0]])
tight = torch.tensor([[7, 8, 9]])
with torch.no_grad():
    padded = model(short, PositionEncoder.padding_mask(short))[0, :3]
    exact = model(tight, PositionEncoder.padding_mask(tight))[0]
print(f"max |delta| between padded and tight: {(padded - exact).abs().max().item():.6f}")

# 3. The 80/10/10 recipe, counted over 10 000 tokens instead of trusted.
masking = MaskingConfig()
batch = torch.randint(max(control_ids("moves")) + 1, cfg.vocab_size, (100, 100))
inputs, labels = apply_masking(batch, masking, cfg.vocab_size, "moves", masking_generator(0))
selected = labels != MMM_IGNORE_INDEX
total = int(selected.sum())
masked = int((inputs[selected] == mask_id("moves")).sum())
kept = int((inputs[selected] == batch[selected]).sum())
print(f"selected {total / batch.numel():.3%} of the tokens (target 15 %)")
# One line in the lesson; split here only so it fits the project's 100-column ruff rule.
shares = f"  <mask> {masked / total:.1%}  kept {kept / total:.1%}"
print(f"{shares}  random {1 - (masked + kept) / total:.1%}")
