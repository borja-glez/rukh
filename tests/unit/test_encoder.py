"""Tests for rukh.models.encoder: shapes, bidirectionality, padding, pooling and the preset."""

from __future__ import annotations

import pytest
import torch

from rukh.models import EncoderConfig, PositionEncoder
from rukh.models.encoder import MMM_IGNORE_INDEX
from rukh.models.squares import SQUARE_TOKENS, SQUARE_VOCAB_SIZE, fen_to_tokens

pytestmark = pytest.mark.unit

TOY = EncoderConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, block=16, dropout=0.0)
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def toy_model(seed: int = 0, **overrides: object) -> PositionEncoder:
    """A deterministic two-layer encoder in eval mode."""
    torch.manual_seed(seed)
    return PositionEncoder(TOY.model_copy(update=overrides)).eval()


def toy_batch(batch: int = 2, seq: int = 8, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, TOY.vocab_size, (batch, seq), generator=generator)


def test_forward_returns_one_hidden_state_per_token() -> None:
    hidden = toy_model()(toy_batch())
    assert hidden.shape == (2, 8, TOY.d_model)


@pytest.mark.parametrize("pos", ["learned", "rope"])
def test_a_later_token_does_change_the_earlier_outputs(pos: str) -> None:
    """The inverse of the decoder's causality test: this model is bidirectional on purpose."""
    model = toy_model(pos=pos)
    idx = toy_batch(seq=12)
    cut = 5
    changed = idx.clone()
    changed[:, cut + 1 :] = (changed[:, cut + 1 :] + 7) % (TOY.vocab_size - 1) + 1
    assert not torch.equal(idx, changed)
    with torch.no_grad():
        base, other = model(idx), model(changed)
    assert not torch.allclose(base[:, : cut + 1], other[:, : cut + 1], atol=1e-5)


def test_padding_does_not_leak_into_the_real_tokens() -> None:
    model = toy_model()
    short = toy_batch(seq=6, seed=2)
    padded = torch.cat([short, torch.zeros((2, 4), dtype=torch.long)], dim=1)
    with torch.no_grad():
        plain = model(short, model.padding_mask(short))
        with_padding = model(padded, model.padding_mask(padded))
    assert torch.allclose(plain, with_padding[:, :6], atol=1e-5)
    # Without the mask the padding is just another token and does change the answer.
    with torch.no_grad():
        unmasked = model(padded)
    assert not torch.allclose(plain, unmasked[:, :6], atol=1e-5)


def test_an_entirely_masked_sequence_is_refused() -> None:
    model = toy_model()
    idx = torch.zeros((2, 6), dtype=torch.long)
    with pytest.raises(ValueError, match="at least one unmasked token"):
        model(idx, model.padding_mask(idx))
    with pytest.raises(ValueError, match=r"\(B, T\)"):
        model(toy_batch(), torch.ones((2, 1, 8), dtype=torch.bool))


def test_mean_pooling_ignores_the_padding() -> None:
    model = toy_model()
    short = toy_batch(seq=6, seed=3)
    padded = torch.cat([short, torch.zeros((2, 5), dtype=torch.long)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model.pool(short, "mean"), model.pool(padded, "mean"), atol=1e-5)
        assert torch.allclose(model.pool(short, "cls"), model.pool(padded, "cls"), atol=1e-5)
        hidden = model(short, model.padding_mask(short))
        assert torch.allclose(model.pool(short, "mean"), hidden.mean(dim=1), atol=1e-5)
        assert torch.allclose(model.pool(short, "cls"), hidden[:, 0], atol=1e-6)
    assert model.pool(short).shape == (2, TOY.d_model)  # mean is the default
    with pytest.raises(ValueError, match="cls"):
        model.pool(short, "first")  # type: ignore[arg-type]


def test_masked_lm_scores_only_the_masked_positions() -> None:
    model = toy_model()
    idx = toy_batch(seq=8, seed=4)
    labels = torch.full_like(idx, MMM_IGNORE_INDEX)
    labels[:, 2] = idx[:, 2]
    logits, loss = model.masked_lm(idx, labels)
    assert logits.shape == (2, 8, TOY.vocab_size)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    assert model.masked_lm(idx)[1] is None

    # Moving a label to another position changes the loss; ignoring everything gives no loss.
    elsewhere = torch.full_like(idx, MMM_IGNORE_INDEX)
    elsewhere[:, 5] = idx[:, 5]
    assert not torch.allclose(loss, model.masked_lm(idx, elsewhere)[1], atol=1e-6)
    assert torch.isnan(model.masked_lm(idx, torch.full_like(idx, MMM_IGNORE_INDEX))[1])
    # ``<pad>`` (0) is a legitimate label here, unlike in the decoder.
    pad_label = torch.full_like(idx, MMM_IGNORE_INDEX)
    pad_label[:, 1] = 0
    assert torch.isfinite(model.masked_lm(idx, pad_label)[1])


def test_the_masked_move_head_is_tied_only_for_moves() -> None:
    moves = toy_model()
    assert moves.mlm_head.weight is moves.tokens.weight
    untied = toy_model(tie_embeddings=False)
    assert untied.mlm_head.weight is not untied.tokens.weight
    torch.manual_seed(0)
    squares = PositionEncoder(EncoderConfig(input="squares", n_layer=2, n_head=4, d_model=32))
    assert squares.mlm_head.weight is not squares.tokens.weight
    assert squares.tokens.num_embeddings == SQUARE_VOCAB_SIZE


def test_the_squares_scheme_has_69_fixed_positions() -> None:
    cfg = EncoderConfig(input="squares", n_layer=2, n_head=4, d_model=32, dropout=0.0)
    assert cfg.seq == SQUARE_TOKENS and cfg.tokens == SQUARE_VOCAB_SIZE
    torch.manual_seed(0)
    model = PositionEncoder(cfg).eval()
    assert model.positions is not None and model.positions.num_embeddings == SQUARE_TOKENS
    idx = torch.tensor([fen_to_tokens(START), fen_to_tokens(f"{START} 30 40")])
    with torch.no_grad():
        assert model(idx).shape == (2, SQUARE_TOKENS, 32)
        assert model.pool(idx, "cls").shape == (2, 32)
    with pytest.raises(ValueError, match="longer than block"):
        model(torch.zeros((1, SQUARE_TOKENS + 1), dtype=torch.long))


def test_same_seed_gives_the_same_weights_and_hidden_states() -> None:
    idx = toy_batch()
    with torch.no_grad():
        a, b, c = toy_model(seed=3)(idx), toy_model(seed=3)(idx), toy_model(seed=4)(idx)
    assert torch.equal(a, b)
    assert not torch.allclose(a, c, atol=1e-4)


def test_preset_size(capsys: pytest.CaptureFixture[str]) -> None:
    counts = {
        scheme: PositionEncoder(EncoderConfig(input=scheme)).num_params(non_embedding=False)
        for scheme in ("moves", "squares")
    }
    with capsys.disabled():
        print()
        for scheme, total in counts.items():
            print(f"encoder preset ({scheme:<7}) {total:>12,} parameters")
    # 8 layers of d=384 are 14,195,712 parameters, plus 779,520 of embedding (2 030 moves) and
    # 76,800 of position table: 15,052,800. The plan's "18M-25M" band was an estimate its own
    # preset (n_layer=8, n_head=6, d_model=384) cannot reach; the preset is what is binding.
    assert 14_000_000 <= counts["moves"] <= 16_000_000
    assert counts["squares"] < counts["moves"]  # 47 square tokens against 2 030 moves


def test_config_rejects_unknown_keys_and_bad_shapes() -> None:
    with pytest.raises(ValueError):
        EncoderConfig(n_layers=3)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="not divisible"):
        EncoderConfig(d_model=100, n_head=8)
    with pytest.raises(ValueError, match="even head dimension"):
        EncoderConfig(d_model=12, n_head=4, pos="rope")
    with pytest.raises(ValueError, match="dropout"):
        EncoderConfig(dropout=1.0)
    with pytest.raises(ValueError, match="block"):
        EncoderConfig(block=0)
    assert EncoderConfig(d_model=64, n_head=4).ff == 256
    assert EncoderConfig(d_model=64, n_head=4, d_ff=128).ff == 128


def test_a_few_steps_of_gradient_descent_reduce_the_loss() -> None:
    torch.manual_seed(5)
    model = PositionEncoder(TOY)
    idx = toy_batch(seq=8, seed=6)
    labels = torch.full_like(idx, MMM_IGNORE_INDEX)
    labels[:, ::2] = idx[:, ::2]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        _, loss = model.masked_lm(idx, labels)
        assert loss is not None
        losses.append(loss.detach().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
