"""``MoveDecoder`` dressed as a ``PreTrainedModel``, so the Hugging Face stack works on it.

The decoder is deliberately plain PyTorch: it is the model of this course and reading it should
not require knowing anybody's framework. But the rest of the ecosystem -- ``peft``, ``trl``'s
trainers, ``AutoModel``, the Hub's inference widgets -- speaks ``PreTrainedModel``, and a wrapper
is a hundred lines whereas re-implementing LoRA, DPO and GRPO is not.

The wrapper **owns** a ``MoveDecoder``; it does not reimplement one. So there is exactly one
definition of the forward pass in this repository, and the test that the two give identical logits
is a test that the wrapping is faithful rather than a second implementation drifting from the
first. The weights are the same objects, which is also why ``from_decoder`` costs nothing.

What this buys, concretely, is the check M4 owes the reader: the hand-written LoRA of
``rukh.models.lora`` and ``peft``'s, on the same weights with the same rank and the same targets,
have to produce the same loss curve. Without the wrapper the two could not be put side by side at
all, and "we implemented LoRA" would rest on the author's word.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from rukh.models.config import DecoderConfig
from rukh.models.decoder import IGNORE_INDEX, MoveDecoder

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel
else:  # pragma: no cover - resolved at import time
    from transformers import PretrainedConfig, PreTrainedModel

from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

__all__ = ["MODEL_TYPE", "RukhConfig", "RukhForCausalLM"]

MODEL_TYPE = "rukh"


class RukhConfig(PretrainedConfig):
    """``DecoderConfig`` in the shape ``transformers`` expects to serialise.

    The fields are the decoder's, spelled the same way, so a ``config.json`` written here and the
    YAML a run was configured with say the same thing. ``transformers`` insists on a few names of
    its own (``vocab_size``, ``model_type``); everything else is left alone rather than renamed
    into ``hidden_size``/``num_hidden_layers``, because a reader who has the decoder open should
    recognise what they are looking at.
    """

    model_type = MODEL_TYPE

    def __init__(
        self,
        vocab_size: int = 2030,
        n_layer: int = 12,
        n_head: int = 8,
        d_model: int = 512,
        block: int = 200,
        dropout: float = 0.0,
        pos: str = "learned",
        tie_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.block = block
        self.dropout = dropout
        self.pos = pos
        self.tie_embeddings = tie_embeddings
        kwargs.setdefault("tie_word_embeddings", tie_embeddings)
        super().__init__(**kwargs)

    @classmethod
    def from_decoder(cls, cfg: DecoderConfig, **kwargs: Any) -> RukhConfig:
        return cls(
            vocab_size=cfg.vocab_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            d_model=cfg.d_model,
            block=cfg.block,
            dropout=cfg.dropout,
            pos=cfg.pos,
            tie_embeddings=cfg.tie_embeddings,
            **kwargs,
        )

    def decoder_config(self) -> DecoderConfig:
        """The ``DecoderConfig`` this config describes.

        Not called ``decoder``: ``transformers`` reads a config attribute of that name as the
        decoder half of an encoder-decoder pair and tries to call ``to_dict`` on it, which turns
        a bound method into a crash the first time anything asks for a generation config.
        """
        return DecoderConfig(
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            d_model=self.d_model,
            block=self.block,
            dropout=self.dropout,
            pos=self.pos,  # type: ignore[arg-type]
            tie_embeddings=self.tie_embeddings,
        )


class RukhForCausalLM(PreTrainedModel, GenerationMixin):
    """A ``PreTrainedModel`` whose whole body is one ``MoveDecoder``.

    ``GenerationMixin`` comes along because ``prepare_inputs_for_generation`` is defined and a
    class that answers it without inheriting the mixin is a model that looks generative and is
    not. Nothing in this project generates through ``.generate()`` -- the sampler of
    ``rukh.infer`` does it with the legality mask, which ``transformers`` knows nothing about --
    but a wrapper that half-implements an interface is worse than one that implements it.
    """

    config_class = RukhConfig
    base_model_prefix = "decoder"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block"]
    _tied_weights_keys = {"decoder.lm_head.weight": "decoder.tokens.weight"}
    """Which name is the copy and which is the storage: the head shares its tensor with the
    embedding table, because ``MoveDecoder`` ties them in its constructor. ``safetensors`` stores
    tensors and not aliases, so ``save_pretrained`` refuses a state dict with two names for one
    tensor unless it is told. This is the same fact ``rukh.train.checkpoint.TIED_SOURCES`` records
    for the project's own checkpoint format."""

    def __init__(self, config: RukhConfig) -> None:
        super().__init__(config)
        self.decoder = MoveDecoder(config.decoder_config())
        # Registers the tied-weight map and the other static properties on the top model. It does
        # not touch the weights in this version of `transformers`, so calling it after building
        # the decoder is safe; skipping it leaves `all_tied_weights_keys` unset and every save or
        # load fails on an attribute the framework expects every model to have.
        self.post_init()

    @classmethod
    def from_decoder(cls, decoder: MoveDecoder, **kwargs: Any) -> RukhForCausalLM:
        """Wrap an existing decoder **without copying its weights**.

        The tensors are shared, so training the wrapper trains the decoder and a checkpoint taken
        from either is the same checkpoint. That is the property that makes the peft comparison
        meaningful: both sides start from literally the same numbers.
        """
        model = cls(RukhConfig.from_decoder(decoder.cfg, **kwargs))
        model.decoder = decoder
        return model

    def get_input_embeddings(self) -> nn.Module:
        return self.decoder.tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.decoder.tokens = value  # type: ignore[assignment]

    def get_output_embeddings(self) -> nn.Module:
        return self.decoder.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,  # noqa: ARG002 - the decoder is always causal
        return_dict: bool | None = None,  # noqa: ARG002 - a dataclass is the only output shape
        **kwargs: Any,
    ) -> CausalLMOutput:
        """Logits, and the decoder's own loss when ``labels`` are given.

        ``labels`` are **already shifted** here, unlike most causal models in ``transformers``.
        The packed stream of this project stores ``(x, y)`` pairs where ``y`` is ``x`` moved one
        step, so shifting again inside the model would teach it to predict the move after next.
        The loss is the decoder's, ``ignore_index=0``, so a run through the wrapper and a run
        through the training loop optimise the same number.
        """
        logits, loss = self.decoder(input_ids, labels)
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, **kwargs: Any
    ) -> dict[str, Any]:
        """Crop to the context window; there is no KV cache to carry between steps."""
        return {"input_ids": input_ids[:, -self.config.block :]}

    @property
    def ignore_index(self) -> int:
        """The id the loss skips: ``<pad>``, id 0, in every scheme of this project."""
        return IGNORE_INDEX
