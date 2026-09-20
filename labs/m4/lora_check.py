"""Check that the LoRA written by hand is LoRA, including against `peft`.

Lab 4 and Lab 5 of M4, in one script and with no checkpoint: it builds its own toy decoder.

Four claims, in order of how easy they are to get wrong:

1. **An untrained adapter is the identity.** `B` starts at zero, so applying LoRA changes nothing
   until a gradient arrives. Any other initialisation moves the weights before the model has
   learned anything, and every comparison against the base starts from a model that is already
   different.
2. **Only the factors train**, and there are exactly `layers x targets x 2 x r x d_model` of them.
3. **Merging is exact.** `W + BA` folded into the weight gives the same logits as the wrapper
   computing it on the fly. That is what lets a 1.6 MB adapter be exported, quantised and served
   as an ordinary model.
4. **It is the same update rule as `peft`'s.** Same base weights, same `A` copied across, same
   optimiser, same batch: the two loss curves have to coincide step by step. If the scaling, the
   side `A` multiplies on, or the zero initialisation of `B` differed, they would separate on the
   first step -- and all three mistakes produce a model that trains and converges anyway.

Run from the repository root:

    uv run python labs/m4/lora_check.py

The fourth check needs the `hf` extra (`uv sync --extra cu128 --extra hf`); without it the script
runs the first three and says so.
"""

import sys

import torch

from rukh.models import DecoderConfig, MoveDecoder, preset
from rukh.models.lora import LoraConfig, apply_lora, lora_modules, merge_lora

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOY = DecoderConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, block=16, dropout=0.0)
R, ALPHA, STEPS = 4, 8, 6


def toy(seed: int = 0) -> MoveDecoder:
    torch.manual_seed(seed)
    return MoveDecoder(TOY)


torch.manual_seed(99)
x = torch.randint(1, 64, (4, 16))
y = torch.randint(1, 64, (4, 16))

# --- 1. an untrained adapter is the identity ---------------------------------
model = toy().eval()
with torch.no_grad():
    before = model(x)[0]
trainable = apply_lora(model, LoraConfig(r=R, alpha=ALPHA, targets=("q", "v")))
with torch.no_grad():
    after = model(x)[0]
print(f"1. identity at step 0: max |delta| = {(before - after).abs().max():.2e}")

# --- 2. the parameter count is the formula -----------------------------------
expected = TOY.n_layer * 2 * 2 * R * TOY.d_model
by_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"2. trainable: {trainable:,} == {by_grad:,} == layers*targets*2*r*d_model = {expected:,}")
medium = preset("medium")
real = medium.n_layer * 2 * 2 * 8 * medium.d_model
print(f"   on `medium` with r=8: {real:,} of 115,120,128 = {100 * real / 115_120_128:.3f} %")
print(f"   as float32: {real * 4:,} bytes = {real * 4 / 1e6:.1f} MB")

# --- 3. merging is exact ------------------------------------------------------
torch.manual_seed(7)
with torch.no_grad():
    for _, adapter in lora_modules(model):
        for b in adapter.b:
            b.normal_(std=0.05)
    adapted = model(x)[0].clone()
merge_lora(model)
with torch.no_grad():
    merged = model(x)[0]
print(f"3. merge is exact: max |delta| = {(adapted - merged).abs().max():.2e}")
print(f"   and no wrappers are left: {len(list(lora_modules(model)))} adapters")

# --- 4. the same update rule as `peft` ---------------------------------------
try:
    import peft

    from rukh.models.hf import RukhForCausalLM
except ImportError as exc:
    print(f"4. skipped: needs the `hf` extra ({exc})")
    raise SystemExit(0) from None

ours = toy()
apply_lora(ours, LoraConfig(r=R, alpha=ALPHA, targets=("attn_out",)))
theirs = peft.get_peft_model(
    RukhForCausalLM.from_decoder(toy()),
    peft.LoraConfig(
        r=R, lora_alpha=ALPHA, lora_dropout=0.0, bias="none", target_modules=["attn.proj"]
    ),
)

# `attn.proj` and not `q`/`v`: `peft` cannot express a per-slice adapter on a fused `qkv`, so over
# there `target_modules=["qkv"]` would be ONE A/B pair of 2304 rows against our one per slice.
# Different parameterisations; comparing them would measure that difference, not the code.
ours_by_layer = dict(lora_modules(ours))
with torch.no_grad():
    for name, module in theirs.named_modules():
        if "default" not in getattr(module, "lora_A", {}):
            continue
        key = name.replace("base_model.model.decoder.", "").replace(".base_layer", "")
        module.lora_A["default"].weight.copy_(ours_by_layer[key].a[0])

mine = torch.optim.SGD([p for p in ours.parameters() if p.requires_grad], lr=0.5)
yours = torch.optim.SGD([p for p in theirs.parameters() if p.requires_grad], lr=0.5)
print("4. step   ours          peft          |delta|")
for step in range(STEPS):
    mine.zero_grad()
    a = ours(x, y)[1]
    a.backward()
    mine.step()

    yours.zero_grad()
    b = theirs(x, labels=y).loss
    b.backward()
    yours.step()
    print(f"   {step:>4d}   {a.item():.8f}    {b.item():.8f}    {abs(a.item() - b.item()):.2e}")
