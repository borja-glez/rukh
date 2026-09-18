"""Tests for rukh.models: shapes, causality, loss masking, RoPE, presets and determinism."""

from __future__ import annotations

import pytest
import torch

from rukh.models import PRESETS, DecoderConfig, MoveDecoder, preset

pytestmark = pytest.mark.unit

TOY = DecoderConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, block=16)


def toy_model(seed: int = 0, **overrides: object) -> MoveDecoder:
    """A deterministic two-layer decoder in eval mode."""
    torch.manual_seed(seed)
    model = MoveDecoder(TOY.model_copy(update=overrides))
    return model.eval()


def toy_batch(batch: int = 2, seq: int = 8, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, TOY.vocab_size, (batch, seq), generator=generator)


def test_logits_have_batch_time_vocab_shape() -> None:
    model = toy_model()
    idx = toy_batch()
    logits, loss = model(idx)
    assert logits.shape == (2, 8, TOY.vocab_size)
    assert loss is None


@pytest.mark.parametrize("pos", ["learned", "rope"])
def test_future_tokens_do_not_change_past_logits(pos: str) -> None:
    model = toy_model(pos=pos)
    idx = toy_batch(seq=12)
    cut = 5
    changed = idx.clone()
    changed[:, cut + 1 :] = (changed[:, cut + 1 :] + 7) % (TOY.vocab_size - 1) + 1
    assert not torch.equal(idx, changed)
    with torch.no_grad():
        base, _ = model(idx)
        other, _ = model(changed)
    assert torch.allclose(base[:, : cut + 1], other[:, : cut + 1], atol=1e-5)
    assert not torch.allclose(base[:, cut + 1 :], other[:, cut + 1 :], atol=1e-5)


@pytest.mark.parametrize("pos", ["learned", "rope"])
def test_rope_and_learned_share_the_output_shape(pos: str) -> None:
    logits, _ = toy_model(pos=pos)(toy_batch())
    assert logits.shape == (2, 8, TOY.vocab_size)


def test_targets_give_a_scalar_loss_that_ignores_padding() -> None:
    model = toy_model()
    idx = toy_batch()
    targets = toy_batch(seed=2)
    _, loss = model(idx, targets)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)

    padded = targets.clone()
    padded[:, 4:] = 0  # <pad>: ignored by the loss
    _, masked_loss = model(idx, padded)
    _, short_loss = model(idx[:, :4], targets[:, :4])
    assert masked_loss is not None and short_loss is not None
    assert torch.allclose(masked_loss, short_loss, atol=1e-5)

    all_pad = torch.zeros_like(targets)
    _, empty = model(idx, all_pad)
    assert empty is not None and torch.isnan(empty)


def test_same_seed_gives_the_same_weights_and_logits() -> None:
    idx = toy_batch()
    with torch.no_grad():
        a, _ = toy_model(seed=3)(idx)
        b, _ = toy_model(seed=3)(idx)
        c, _ = toy_model(seed=4)(idx)
    assert torch.equal(a, b)
    assert not torch.allclose(a, c, atol=1e-4)


def test_next_logits_match_the_last_step_of_forward() -> None:
    model = toy_model()
    idx = toy_batch(seq=9)
    with torch.no_grad():
        full, _ = model(idx)
    assert torch.allclose(model.next_logits(idx), full[:, -1], atol=1e-6)


def test_next_logits_crop_sequences_longer_than_the_block() -> None:
    model = toy_model()
    long_idx = toy_batch(seq=TOY.block + 5)
    assert model.next_logits(long_idx).shape == (2, TOY.vocab_size)
    with pytest.raises(ValueError, match="longer than block"):
        model(long_idx)


def test_embeddings_are_tied_and_can_be_untied() -> None:
    tied = toy_model()
    assert tied.lm_head.weight is tied.tokens.weight
    untied = toy_model(tie_embeddings=False)
    assert untied.lm_head.weight is not untied.tokens.weight


def test_preset_sizes(capsys: pytest.CaptureFixture[str]) -> None:
    counts = {}
    for name in PRESETS:
        model = MoveDecoder(preset(name))
        counts[name] = (model.num_params(), model.num_params(non_embedding=False))
    with capsys.disabled():
        print()
        for name, (non_embedding, total) in counts.items():
            print(f"{name:<7} {non_embedding:>12,} non-embedding  {total:>12,} total")
    assert 38_000_000 <= counts["small"][0] <= 45_000_000
    assert counts["tiny"][0] < counts["small"][0] < counts["medium"][0]


def test_preset_copies_are_independent() -> None:
    first = preset("tiny")
    first.dropout = 0.5
    assert preset("tiny").dropout == 0.0
    assert PRESETS["tiny"].dropout == 0.0
    with pytest.raises(ValueError, match="unknown preset"):
        preset("enormous")


def test_config_rejects_unknown_keys_and_bad_shapes() -> None:
    with pytest.raises(ValueError):
        DecoderConfig(n_layers=3)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="not divisible"):
        DecoderConfig(d_model=100, n_head=8)
    with pytest.raises(ValueError, match="even head dimension"):
        DecoderConfig(d_model=12, n_head=4, pos="rope")


def test_d_ff_defaults_to_four_times_d_model() -> None:
    assert DecoderConfig(d_model=64, n_head=4).ff == 256
    assert DecoderConfig(d_model=64, n_head=4, d_ff=128).ff == 128
    assert MoveDecoder(TOY.model_copy(update={"d_ff": 48})).blocks[0].mlp.fc.out_features == 48


def test_a_few_steps_of_gradient_descent_reduce_the_loss() -> None:
    torch.manual_seed(5)
    model = MoveDecoder(TOY)
    idx = toy_batch(seq=8, seed=6)
    targets = torch.roll(idx, -1, dims=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        _, loss = model(idx, targets)
        assert loss is not None
        losses.append(loss.detach().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
