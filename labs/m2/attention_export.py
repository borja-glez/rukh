"""Export the attention weights of one short game to artifacts/web/attention.json."""

import json
from datetime import UTC, datetime
from pathlib import Path

import chess
import torch
from torch.nn import functional as F

from rukh.models.decoder import apply_rope
from rukh.tokenize.uci_vocab import UciTokenizer
from rukh.train import load_model

CKPT = Path("checkpoints/small/best.pt")
OUT = Path("artifacts/web/attention.json")
# Legal's mate: short, famous and every move is easy to follow in the heat map.
MOVES = "e2e4 e7e5 g1f3 b8c6 f1c4 d7d6 b1c3 c8g4 f3e5 g4d1 c4f7 e8e7 c3d5".split()

model, _ = load_model(CKPT)
tok = UciTokenizer()
board = chess.Board()
for uci in MOVES:  # fail loudly if the line is not legal
    board.push(chess.Move.from_uci(uci))

ids = [tok.bos_id, tok.vocab["<w1800>"], tok.vocab["<b1800>"]]
ids += [tok.vocab[uci] for uci in MOVES]
labels = ["<bos>", "<w1800>", "<b1800>", *MOVES]
idx = torch.tensor([ids], dtype=torch.long)

captured: list[torch.Tensor] = []


def hook(module, args, output):  # noqa: ARG001 - torch hook signature
    captured.append(output.detach())


handles = [block.attn.qkv.register_forward_hook(hook) for block in model.blocks]
with torch.no_grad():
    model(idx)
for handle in handles:
    handle.remove()

cfg = model.cfg
weights = []
mask = torch.ones(len(ids), len(ids), dtype=torch.bool).tril()
for _layer, qkv in enumerate(captured):
    q, k, _ = qkv.split(cfg.d_model, dim=2)
    shape = (1, len(ids), cfg.n_head, cfg.head_dim)
    q = q.view(shape).transpose(1, 2)
    k = k.view(shape).transpose(1, 2)
    if cfg.pos == "rope":
        q = apply_rope(q, model.rope_cos, model.rope_sin)
        k = apply_rope(k, model.rope_cos, model.rope_sin)
    scores = (q @ k.transpose(-2, -1)) / (cfg.head_dim**0.5)
    probs = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)[0]
    weights.append([[[round(v, 4) for v in row] for row in head] for head in probs.tolist()])

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(
        {
            "schema": "rukh-attention/1",
            "game": {"moves": labels},
            "layers": cfg.n_layer,
            "heads": cfg.n_head,
            "weights": weights,
            "meta": {
                "model": "rukh-small",
                "checkpoint": str(CKPT),
                "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        }
    ),
    encoding="utf-8",
)
print(f"wrote {OUT} ({cfg.n_layer} layers x {cfg.n_head} heads x {len(ids)}^2)")
