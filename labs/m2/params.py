"""Parameter count of a MoveDecoder, block by block, against the closed-form formula."""

from rukh.models import MoveDecoder, preset


def formula(cfg) -> int:
    """What the architecture says it should be, computed by hand."""
    d, ff, v = cfg.d_model, cfg.ff, cfg.vocab_size
    per_block = (
        2 * d  # ln1
        + (d * 3 * d + 3 * d)  # qkv
        + (d * d + d)  # attn out projection
        + 2 * d  # ln2
        + (d * ff + ff)  # mlp in
        + (ff * d + d)  # mlp out
    )
    total = v * d + per_block * cfg.n_layer + 2 * d  # tokens + blocks + final ln
    if cfg.pos == "learned":
        total += cfg.block * d
    if not cfg.tie_embeddings:
        total += v * d
    return total


for name in ("tiny", "small", "medium"):
    cfg = preset(name)
    model = MoveDecoder(cfg)
    counted = sum(p.numel() for p in model.parameters())
    groups = {
        "tokens": model.tokens.weight.numel(),
        "positions": model.positions.weight.numel() if model.positions is not None else 0,
        "blocks": sum(p.numel() for p in model.blocks.parameters()),
        "ln_f": sum(p.numel() for p in model.ln_f.parameters()),
        "lm_head (tied)": 0 if cfg.tie_embeddings else model.lm_head.weight.numel(),
    }
    print(f"== {name}: {cfg.n_layer} layers, d={cfg.d_model}, {cfg.n_head} heads ==")
    for key, value in groups.items():
        print(f"  {key:<16} {value:>12,}")
    print(f"  {'total':<16} {counted:>12,}   formula {formula(cfg):>12,}")
    print(f"  {'non-embedding':<16} {model.num_params():>12,}")
    assert counted == formula(cfg), f"{name}: the formula does not match the model"
