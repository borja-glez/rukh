"""How many directions a rank-8 adapter really moves, measured with an SVD.

Lab 5b of M4. "The correction has to pass through eight dimensions" is easy to write and easy to
believe without checking. The check is one line of linear algebra: the singular values of
``delta W = (alpha / r) B A`` are the lengths of the directions that correction can push along, and
there are exactly ``r`` of them that are not zero. Next to the singular values of ``W`` itself --
seven hundred and sixty-eight of them, all non-zero -- the picture is the whole method.

Run from the repository root:

    uv run python labs/m4/lora_spectrum.py --adapter checkpoints/lora-e4-20260920-160133
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from rukh.models.lora import load_adapter, lora_modules
from rukh.train import load_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", default="checkpoints/medium-v4-20260919-174623/best.pt", type=Path)
parser.add_argument("--adapter", default="checkpoints/lora-e4-20260920-160133", type=Path)
parser.add_argument("--layer", default=0, type=int, help="Block to read the matrices from.")
parser.add_argument("--keep", default=24, type=int, help="Singular values to write out.")
parser.add_argument("--out", default="artifacts/web/lora-spectrum.json", type=Path)
args = parser.parse_args()

model = load_model(args.ckpt)[0].eval()
cfg = load_adapter(model, args.adapter / "adapter.safetensors")
adapters = dict(lora_modules(model))
name = f"blocks.{args.layer}.attn.qkv"
adapter = adapters[name]

# The query slice of the fused projection: the first `d_model` output rows.
start, stop = adapter.slices[0]
with torch.no_grad():
    base = adapter.base.weight[start:stop].float()
    delta = (cfg.scale * (adapter.b[0].float() @ adapter.a[0].float())).float()
    base_sv = torch.linalg.svdvals(base)
    delta_sv = torch.linalg.svdvals(delta)

nonzero = int((delta_sv > delta_sv.max() * 1e-6).sum())
print(f"{name} query slice: W is {tuple(base.shape)}, delta W is {tuple(delta.shape)}")
print(f"  r = {cfg.r}, alpha = {cfg.alpha}, scale = {cfg.scale}")
print(
    f"  non-zero singular values: W {int((base_sv > base_sv.max() * 1e-6).sum())}, "
    f"delta W {nonzero}"
)
print(f"  ||delta W||_F / ||W||_F = {torch.linalg.norm(delta) / torch.linalg.norm(base):.4f}")
print("  first singular values of delta W:", [round(float(v), 4) for v in delta_sv[: cfg.r + 2]])

payload = {
    "adapter": args.adapter.name,
    "module": name,
    "r": cfg.r,
    "alpha": cfg.alpha,
    "scale": cfg.scale,
    "shape": list(base.shape),
    "nonzero_delta": nonzero,
    "relative_norm": float(torch.linalg.norm(delta) / torch.linalg.norm(base)),
    "base": [float(v) for v in base_sv[: args.keep]],
    "delta": [float(v) for v in delta_sv[: args.keep]],
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
print(f"\nwrote {args.out}")
