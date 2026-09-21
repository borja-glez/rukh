"""Tests for rukh.train.dpo: the loss, the batching and what gets dropped."""

from __future__ import annotations

import polars as pl
import pytest
import torch

from rukh.tokenize.uci_vocab import UciTokenizer
from rukh.train.dpo import DpoConfig, batches, dpo_loss, encode_pairs

pytestmark = pytest.mark.unit


@pytest.fixture
def tok() -> UciTokenizer:
    return UciTokenizer()


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "prefix": pl.String,
            "chosen": pl.String,
            "rejected": pl.String,
            "white_elo": pl.Int32,
            "black_elo": pl.Int32,
        },
    )


def test_encode_builds_the_header_and_the_prefix(tok: UciTokenizer) -> None:
    rows = encode_pairs(
        _frame(
            [
                {
                    "prefix": "e2e4 e7e5",
                    "chosen": "g1f3",
                    "rejected": "h2h4",
                    "white_elo": 1850,
                    "black_elo": 2250,
                }
            ]
        ),
        tok,
        block=200,
    )
    assert len(rows) == 1
    ids, chosen, rejected = rows[0]
    assert ids[0] == tok.bos_id
    assert ids[1] == tok.vocab["<w1800>"] and ids[2] == tok.vocab["<b2200>"]
    assert ids[3:] == [tok.vocab["e2e4"], tok.vocab["e7e5"]]
    assert chosen == tok.vocab["g1f3"] and rejected == tok.vocab["h2h4"]


def test_encode_drops_what_it_cannot_represent(tok: UciTokenizer) -> None:
    """A prompt that does not fit is dropped, never truncated.

    Cutting the front of a prefix would hand the model a position that never occurred in the
    game, and it would look like a valid example.
    """
    long_prefix = " ".join(["e2e4"] * 300)
    rows = encode_pairs(
        _frame(
            [
                {
                    "prefix": long_prefix,
                    "chosen": "g1f3",
                    "rejected": "h2h4",
                    "white_elo": 1800,
                    "black_elo": 1800,
                },
                {
                    "prefix": "e2e4",
                    "chosen": "zzzz",
                    "rejected": "h2h4",
                    "white_elo": 1800,
                    "black_elo": 1800,
                },
                {
                    "prefix": "e2e4",
                    "chosen": "g1f3",
                    "rejected": "g1f3",
                    "white_elo": 1800,
                    "black_elo": 1800,
                },
            ]
        ),
        tok,
        block=200,
    )
    assert rows == []


def test_batches_pad_and_mark_where_each_prompt_ends(tok: UciTokenizer) -> None:
    rows = encode_pairs(
        _frame(
            [
                {
                    "prefix": "e2e4",
                    "chosen": "e7e5",
                    "rejected": "h7h5",
                    "white_elo": 1800,
                    "black_elo": 1800,
                },
                {
                    "prefix": "e2e4 e7e5 g1f3",
                    "chosen": "b8c6",
                    "rejected": "h7h5",
                    "white_elo": 1800,
                    "black_elo": 1800,
                },
            ]
        ),
        tok,
        block=200,
    )
    batch = next(iter(batches(rows, 8, tok.pad_id, shuffle=False, seed=0)))
    assert batch.tokens.shape == (2, 6)
    assert batch.last.tolist() == [3, 5]
    # Everything past each prompt is padding, so the model never reads another row's moves.
    assert batch.tokens[0, 4:].eq(tok.pad_id).all()


def test_loss_falls_when_the_policy_prefers_the_better_move() -> None:
    """The objective has to reward exactly one thing: ordering the pair correctly."""
    vocab = 8
    reference = torch.log_softmax(torch.zeros(1, vocab), dim=-1)
    batch = type(
        "B",
        (),
        {
            "chosen": torch.tensor([1]),
            "rejected": torch.tensor([2]),
            "tokens": torch.zeros(1, 1, dtype=torch.long),
            "last": torch.tensor([0]),
        },
    )()

    good = torch.full((1, vocab), -5.0)
    good[0, 1] = 5.0
    bad = torch.full((1, vocab), -5.0)
    bad[0, 2] = 5.0
    good_loss, good_margin, good_acc = dpo_loss(
        torch.log_softmax(good, -1), reference, batch, beta=0.1
    )
    bad_loss, bad_margin, bad_acc = dpo_loss(torch.log_softmax(bad, -1), reference, batch, beta=0.1)

    assert good_loss < bad_loss
    assert good_margin > 0 > bad_margin
    assert good_acc == 1.0 and bad_acc == 0.0


def test_config_rejects_a_negative_beta() -> None:
    with pytest.raises(ValueError):
        DpoConfig(beta=0.0)


def test_max_pairs_samples_the_whole_file_with_the_seed() -> None:
    """The off-policy arm is matched to the on-policy count here, not by a script outside git."""
    from rukh.train.dpo import limit_pairs

    frame = pl.DataFrame({"phase": ["opening"] * 6 + ["endgame"] * 6, "i": list(range(12))})
    same = limit_pairs(frame, DpoConfig())
    assert same.height == 12
    cut = limit_pairs(frame, DpoConfig(max_pairs=4, seed=7))
    assert cut.height == 4
    assert cut["i"].to_list() == limit_pairs(frame, DpoConfig(max_pairs=4, seed=7))["i"].to_list()
    assert cut["i"].to_list() != [0, 1, 2, 3]  # not the head of the file
    assert limit_pairs(frame, DpoConfig(max_pairs=50)).height == 12
    with pytest.raises(ValueError):
        DpoConfig(max_pairs=0)
