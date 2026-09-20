"""Tests for rukh.models.lora: the hand-written adapters and what makes them LoRA.

The point of writing LoRA instead of importing it is that every claim the method makes can be
checked here: it starts as the identity, it trains only the factors, folding it back gives the
same model, and the file it produces is three orders of magnitude smaller than the weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rukh.models import DecoderConfig, MoveDecoder, preset
from rukh.models.lora import (
    LoraConfig,
    LoRALinear,
    adapter_params,
    apply_lora,
    load_adapter,
    lora_modules,
    merge_lora,
    save_adapter,
)

pytestmark = pytest.mark.unit


def _model(seed: int = 0) -> MoveDecoder:
    torch.manual_seed(seed)
    return MoveDecoder(DecoderConfig(n_layer=2, n_head=2, d_model=16, block=8, dropout=0.0)).eval()


def _idx() -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randint(1, 100, (2, 8))


def test_an_untrained_adapter_is_the_identity() -> None:
    """`B` starts at zero, so applying LoRA changes nothing until a gradient arrives.

    Any other initialisation would move the model before it learned anything, and every
    comparison against the base would start from a model that is already different.
    """
    idx = _idx()
    model = _model()
    with torch.no_grad():
        before, _ = model(idx)
    apply_lora(model, LoraConfig(r=4))
    with torch.no_grad():
        after, _ = model(idx)
    torch.testing.assert_close(before, after)


def test_only_the_factors_are_trainable() -> None:
    model = _model()
    trainable = apply_lora(model, LoraConfig(r=4, targets=("q", "v")))
    by_grad = [p for p in model.parameters() if p.requires_grad]
    assert trainable == sum(p.numel() for p in by_grad)
    assert {id(p) for p in by_grad} == {id(p) for p in adapter_params(model)}


def test_trainable_count_is_the_formula_on_the_card() -> None:
    """`2 * r * d_model` per adapted `d_model`-wide matrix, per layer, and nothing else."""
    cfg = LoraConfig(r=4, targets=("q", "v"))
    model = _model()
    trainable = apply_lora(model, cfg)
    n_layer, d_model = model.cfg.n_layer, model.cfg.d_model
    assert trainable == n_layer * len(cfg.targets) * 2 * cfg.r * d_model


def test_the_fused_projection_gives_each_target_its_own_factors() -> None:
    """Query and value are two slices of one `qkv` matrix, not one adapter over both.

    Sharing a single `A` between them would be cheaper and would not be LoRA: each adapted
    matrix is supposed to get its own low-rank subspace.
    """
    model = _model()
    apply_lora(model, LoraConfig(r=4, targets=("q", "v")))
    adapters = dict(lora_modules(model))
    assert set(adapters) == {"blocks.0.attn.qkv", "blocks.1.attn.qkv"}
    qkv = adapters["blocks.0.attn.qkv"]
    d = model.cfg.d_model
    assert qkv.slices == ((0, d), (2 * d, 3 * d))  # query and value, key untouched
    assert len(qkv.a) == len(qkv.b) == 2
    assert not torch.equal(qkv.a[0], qkv.a[1])


def test_the_key_slice_never_moves_when_only_q_and_v_are_adapted() -> None:
    model = _model()
    apply_lora(model, LoraConfig(r=4, targets=("q", "v")))
    d = model.cfg.d_model
    for _, adapter in lora_modules(model):
        with torch.no_grad():
            for b in adapter.b:
                b.normal_()
        weight = adapter.merged_weight()
        torch.testing.assert_close(weight[d : 2 * d], adapter.base.weight[d : 2 * d])
        assert not torch.allclose(weight[:d], adapter.base.weight[:d])


def test_merging_reproduces_the_adapted_model_exactly() -> None:
    """`W + BA` folded into the weight has to give the same logits as the wrapper computing it.

    This is the property that lets an adapter be published as a small file and still be
    exported, quantised and served as an ordinary model.
    """
    idx = _idx()
    model = _model()
    apply_lora(model, LoraConfig(r=4, alpha=8, targets=("q", "v")))
    torch.manual_seed(7)
    with torch.no_grad():
        for _, adapter in lora_modules(model):
            for b in adapter.b:
                b.normal_(std=0.05)
        adapted, _ = model(idx)
    assert merge_lora(model) == model.cfg.n_layer
    assert not list(lora_modules(model))
    with torch.no_grad():
        merged, _ = model(idx)
    torch.testing.assert_close(adapted, merged, rtol=1e-5, atol=1e-6)


def test_a_merged_model_has_the_state_dict_of_a_plain_decoder() -> None:
    model = _model()
    keys_before = set(model.state_dict())
    apply_lora(model, LoraConfig(r=4))
    assert "blocks.0.attn.qkv.base.weight" in model.state_dict()
    merge_lora(model)
    assert set(model.state_dict()) == keys_before


def test_adapter_round_trips_through_safetensors(tmp_path: Path) -> None:
    idx = _idx()
    cfg = LoraConfig(r=4, alpha=8, targets=("q", "v"))
    trained = _model()
    apply_lora(trained, cfg)
    torch.manual_seed(11)
    with torch.no_grad():
        for _, adapter in lora_modules(trained):
            for b in adapter.b:
                b.normal_(std=0.05)
        expected, _ = trained(idx)

    path = save_adapter(trained, tmp_path / "adapter.safetensors", cfg)
    assert path.with_name("adapter_config.json").is_file()

    restored = _model()
    loaded = load_adapter(restored, path)
    assert loaded == cfg
    with torch.no_grad():
        got, _ = restored(idx)
    torch.testing.assert_close(expected, got)


def test_the_adapter_is_a_rounding_error_next_to_the_published_model() -> None:
    """The number the whole method exists for, on the model that is actually published.

    The toy decoder of these tests is 39 k parameters and two thirds of that is the embedding
    table, so its ratio says nothing. `medium` is the size the claim is about: r=8 on query and
    value across 16 layers of `d_model` 768 is 393 216 numbers, 0.34 % of 115 M -- 1 572 864 bytes,
    the 1.6 MB file a reader downloads instead of a second copy of the weights.
    """
    cfg = LoraConfig(r=8, alpha=16, targets=("q", "v"))
    medium = preset("medium")
    adapter = medium.n_layer * len(cfg.targets) * 2 * cfg.r * medium.d_model
    published_params = 115_120_128  # `medium`, as every card and report of the project says
    assert adapter == 393_216
    assert adapter / published_params < 0.004
    assert adapter * 4 == 1_572_864  # bytes in float32: 1.6 MB, the size of the published file


def test_gradients_reach_the_factors_and_nothing_else() -> None:
    model = _model()
    apply_lora(model, LoraConfig(r=4, targets=("q", "v")))
    idx = _idx()
    _, loss = model(idx, idx)
    assert loss is not None
    loss.backward()
    assert all(p.grad is not None for p in adapter_params(model))
    assert model.tokens.weight.grad is None
    for _, adapter in lora_modules(model):
        assert adapter.base.weight.grad is None


def test_targets_are_validated() -> None:
    with pytest.raises(ValueError, match="unknown LoRA targets"):
        apply_lora(_model(), LoraConfig(targets=("qq",)))
    with pytest.raises(ValueError, match="at least one"):
        apply_lora(_model(), LoraConfig(targets=()))


def test_mlp_targets_use_the_feed_forward_width() -> None:
    """`mlp.fc` is `ff` wide, not `d_model`; a slice computed in the wrong unit would truncate."""
    model = _model()
    apply_lora(model, LoraConfig(r=2, targets=("mlp_in", "mlp_out")))
    adapters = dict(lora_modules(model))
    assert adapters["blocks.0.mlp.fc"].slices == ((0, model.cfg.ff),)
    assert adapters["blocks.0.mlp.proj"].slices == ((0, model.cfg.d_model),)


def test_scale_follows_alpha_over_r() -> None:
    base = torch.nn.Linear(4, 4)
    adapter = LoRALinear(base, r=2, alpha=8, dropout=0.0, slices=((0, 4),))
    assert adapter.scale == 4.0


def test_the_factors_are_built_where_the_weight_they_correct_lives() -> None:
    """`apply_lora` runs after the model has been moved to its device.

    The training loop loads the checkpoint first -- loading needs the module names of a plain
    decoder -- so by the time the adapters go on, the weights are already on the GPU. A factor
    created on the CPU by default meets a CUDA activation on the first forward and the run dies
    there. Every other test in this file runs on the CPU, where the bug cannot be seen; this one
    checks the property that makes it impossible instead of the device it happens to run on.
    """
    model = _model()
    base = model.blocks[0].attn.qkv.weight
    apply_lora(model, LoraConfig(r=4, targets=("q", "v")))
    for _, adapter in lora_modules(model):
        for factor in [*adapter.a, *adapter.b]:
            assert factor.device == base.device
            assert factor.dtype == base.dtype
