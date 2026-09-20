"""Tests for rukh.models.hf: the wrapper, and the check that our LoRA is LoRA.

The second half is the one that matters for M4. Writing LoRA by hand is only worth the pages it
takes in the lesson if the thing written is the thing the papers describe, and the way to know
that is to put it next to the reference implementation on the same weights and watch the two
optimise the same number. Without this test "we implemented LoRA" would rest on the author's word.

The Hugging Face stack is an optional extra (`uv sync --extra hf`): a CPU CI job with a
three-minute budget should not install it to run P0-P3. These tests skip when it is absent and are
run locally and in the extended job.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rukh.models import DecoderConfig, MoveDecoder
from rukh.models.lora import LoraConfig, apply_lora, lora_modules

transformers = pytest.importorskip("transformers", reason="needs the `hf` extra")
peft = pytest.importorskip("peft", reason="needs the `hf` extra")

from rukh.models.hf import RukhConfig, RukhForCausalLM  # noqa: E402

pytestmark = pytest.mark.unit

TOY = DecoderConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, block=16, dropout=0.0)


def _decoder(seed: int = 0) -> MoveDecoder:
    torch.manual_seed(seed)
    return MoveDecoder(TOY)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(99)
    x = torch.randint(1, 64, (4, 16))
    y = torch.randint(1, 64, (4, 16))
    return x, y


def test_the_wrapper_and_the_decoder_give_the_same_logits() -> None:
    """One definition of the forward pass, wrapped -- not a second one that can drift."""
    decoder = _decoder().eval()
    model = RukhForCausalLM.from_decoder(decoder).eval()
    x, _ = _batch()
    with torch.no_grad():
        torch.testing.assert_close(decoder(x)[0], model(x).logits)


def test_wrapping_shares_the_weights_rather_than_copying_them() -> None:
    decoder = _decoder()
    model = RukhForCausalLM.from_decoder(decoder)
    assert model.decoder is decoder
    assert model.get_input_embeddings().weight.data_ptr() == decoder.tokens.weight.data_ptr()


def test_the_loss_is_the_decoder_s_own() -> None:
    """Same number, so a run through `SFTTrainer` optimises what the training loop optimises."""
    decoder = _decoder().eval()
    model = RukhForCausalLM.from_decoder(decoder).eval()
    x, y = _batch()
    with torch.no_grad():
        expected = decoder(x, y)[1]
        got = model(x, labels=y).loss
    assert expected is not None
    torch.testing.assert_close(expected, got)


def test_labels_are_not_shifted_again() -> None:
    """The packed stream already stores `y` one step ahead of `x` (`rukh.tokenize.loader`).

    Every causal model in `transformers` shifts labels inside `forward`; doing that here would
    train the model to predict the move *after* next, and the loss would still look plausible.
    """
    decoder = _decoder().eval()
    model = RukhForCausalLM.from_decoder(decoder).eval()
    x, _ = _batch()
    shifted = torch.roll(x, shifts=-1, dims=1)
    with torch.no_grad():
        through_wrapper = model(x, labels=shifted).loss
        by_hand = decoder(x, shifted)[1]
    torch.testing.assert_close(through_wrapper, by_hand)


def test_the_config_round_trips_through_disk(tmp_path: Path) -> None:
    decoder = _decoder().eval()
    model = RukhForCausalLM.from_decoder(decoder).eval()
    model.save_pretrained(tmp_path)
    reloaded = RukhForCausalLM.from_pretrained(tmp_path).eval()
    assert reloaded.config.decoder_config() == TOY
    x, _ = _batch()
    with torch.no_grad():
        torch.testing.assert_close(model(x).logits, reloaded(x).logits)


def test_the_config_is_not_mistaken_for_an_encoder_decoder_pair() -> None:
    """`transformers` reads a config attribute called `decoder` as the other half of a pair.

    It calls `to_dict()` on it, so a method with that name crashes the first time anything asks
    for a generation config -- which is anything that touches `.generate()`.
    """
    config = RukhConfig.from_decoder(TOY)
    assert not hasattr(config, "decoder") or not callable(getattr(config, "decoder", None))
    transformers.GenerationConfig.from_model_config(config)


def _peft_config(r: int, alpha: int, targets: list[str]) -> peft.LoraConfig:
    return peft.LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
    )


def test_peft_counts_the_same_trainable_parameters_as_the_hand_written_lora() -> None:
    """Same rank, same matrices: the arithmetic has to agree before the curves can."""
    ours = _decoder()
    mine = apply_lora(ours, LoraConfig(r=4, alpha=8, targets=("attn_out",)))

    theirs = RukhForCausalLM.from_decoder(_decoder())
    adapted = peft.get_peft_model(theirs, _peft_config(4, 8, ["attn.proj"]))
    reference = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    assert mine == reference


def test_the_hand_written_lora_and_peft_optimise_the_same_number() -> None:
    """The check that makes the chapter honest.

    Both sides start from the same base weights and are given the same `A` (copied across, so the
    comparison is of the maths and not of two random draws), then take the same optimiser steps on
    the same batch. If the update rule differed at all -- the scaling, which side `A` multiplies,
    whether `B` starts at zero -- the losses would separate immediately.
    """
    x, y = _batch()
    r, alpha, steps = 4, 8, 6

    ours = _decoder()
    apply_lora(ours, LoraConfig(r=r, alpha=alpha, targets=("attn_out",)))
    mine = [p for p in ours.parameters() if p.requires_grad]

    theirs = RukhForCausalLM.from_decoder(_decoder())
    adapted = peft.get_peft_model(theirs, _peft_config(r, alpha, ["attn.proj"]))

    # Copy our `A` into theirs so both start from the same point; `B` is zero on both sides.
    ours_by_layer = {name: module for name, module in lora_modules(ours)}
    with torch.no_grad():
        for name, module in adapted.named_modules():
            if not hasattr(module, "lora_A") or "default" not in getattr(module, "lora_A", {}):
                continue
            key = name.replace("base_model.model.decoder.", "").replace(".base_layer", "")
            module.lora_A["default"].weight.copy_(ours_by_layer[key].a[0])

    mine_opt = torch.optim.SGD(mine, lr=0.5)
    their_opt = torch.optim.SGD([p for p in adapted.parameters() if p.requires_grad], lr=0.5)

    my_curve: list[float] = []
    their_curve: list[float] = []
    for _ in range(steps):
        mine_opt.zero_grad()
        loss = ours(x, y)[1]
        assert loss is not None
        loss.backward()
        mine_opt.step()
        my_curve.append(float(loss.detach()))

        their_opt.zero_grad()
        other = adapted(x, labels=y).loss
        other.backward()
        their_opt.step()
        their_curve.append(float(other.detach()))

    assert my_curve[-1] < my_curve[0]  # it is training at all
    for step, (a, b) in enumerate(zip(my_curve, their_curve, strict=True)):
        assert a == pytest.approx(b, abs=1e-5), f"step {step}: {a} vs {b}"


def test_peft_leaves_the_base_weights_frozen_exactly_as_ours_does() -> None:
    theirs = RukhForCausalLM.from_decoder(_decoder())
    adapted = peft.get_peft_model(theirs, _peft_config(4, 8, ["attn.proj"]))
    trainable = {name for name, p in adapted.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in name for name in trainable)
