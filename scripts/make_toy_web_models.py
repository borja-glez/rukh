"""Regenerate the toy ONNX files the demo's E2E suite plays against.

They are versioned in `rukh-web/public/test/` so the browser tests walk the real worker path --
download, contract check, session, inference, style swap -- without fetching hundreds of megabytes
from the Hub. They are written by the very functions that write the published models, so they carry
the same `rukh_*` metadata and come out of the same exporter (D-031).

Three files, and the third is what makes the style selector testable:

* `toy-decoder.onnx`      one layer, `d_model=8`, the real 2 030-token vocabulary, `block` 200;
* `toy-decoder-lora.onnx` the same decoder exported with its LoRA factors as inputs;
* `toy-lora.bin`          an adapter for it, of the size that graph declares.

The adapter is **not** trained: it is a fixed pseudo-random `B` with a seed, which is enough for
the only thing the E2E can check without a real corpus -- that loading it changes the move the
model plays, and that clearing it puts the old move back.

Run from the repository root:

    uv run python scripts/make_toy_web_models.py
"""

import argparse
import sys
from pathlib import Path

import torch

from rukh.export import export_onnx
from rukh.export.adapter import browser_lora, export_adaptable_onnx, write_web_adapter
from rukh.models import DecoderConfig, MoveDecoder
from rukh.models.lora import LoraConfig, apply_lora, lora_modules
from rukh.tokenize.uci_vocab import UciTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The vocabulary and the context are the real ones: those two are what the browser checks.
TOY = DecoderConfig(vocab_size=len(UciTokenizer().ids), n_layer=1, n_head=2, d_model=8, block=200)
LORA = LoraConfig(r=2, alpha=4, targets=("q", "v"))
SEED = 31

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="../rukh-web/public/test", type=Path)
parser.add_argument("--seq-len", default=8, type=int, help="Example length the exporter traces.")
parser.add_argument(
    "--plain",
    action="store_true",
    help="Also rewrite `toy-decoder.onnx`. Off by default: the committed one was written with "
    "another seed, and replacing it changes the moves every browser test expects.",
)
args = parser.parse_args()

out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)


def toy() -> MoveDecoder:
    torch.manual_seed(SEED)
    return MoveDecoder(TOY).eval()


if args.plain:
    plain = export_onnx(toy(), out / "toy-decoder.onnx", seq_len=args.seq_len)
    print(f"{plain.path}  {plain.bytes:,} bytes  ({plain.exporter}, opset {plain.opset})")

# `browser_lora` is what the demo's real exports use: the `alpha / r` lives in the adapter file,
# not in the graph, so a graph can take an adapter of any rank without anyone matching two numbers.
adaptable = export_adaptable_onnx(
    toy(), out / "toy-decoder-lora.onnx", browser_lora(LORA), seq_len=args.seq_len
)
print(f"{adaptable.path}  {adaptable.bytes:,} bytes")
print(f"  lora_a {adaptable.metadata['rukh_adapter_shape_a']}")
print(f"  lora_b {adaptable.metadata['rukh_adapter_shape_b']}")

adapted = toy()
apply_lora(adapted, LORA)
torch.manual_seed(SEED + 1)
with torch.no_grad():
    for _, module in lora_modules(adapted):
        for b in module.b:
            # Big enough that the argmax of a one-layer toy actually moves; a style adapter on a
            # real model is a correction, here it only has to be visible.
            b.normal_(std=1.5)
binary = write_web_adapter(adapted, out / "toy-lora.bin", base_repo=None)
print(f"{binary}  {binary.stat().st_size:,} bytes")

# What the E2E asserts: the two files answer differently for the same position.
tok = UciTokenizer()
idx = torch.tensor([[tok.vocab["<bos>"], tok.vocab["<w1800>"], tok.vocab["<b1800>"]]])
with torch.no_grad():
    base_move = int(toy()(idx)[0][0, -1].argmax())
    adapted_move = int(adapted(idx)[0][0, -1].argmax())
print(f"\nfirst move: base {tok.ids[base_move]}, with the toy adapter {tok.ids[adapted_move]}")
if base_move == adapted_move:
    raise SystemExit("the toy adapter does not change the first move; raise the std and re-run")
