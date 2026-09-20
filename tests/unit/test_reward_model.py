"""The reward model: what Bradley-Terry does and does not promise.

A preference loss only ever sees *differences*, which has a consequence worth a test of its own:
the scale has no meaning, and a model whose answer changes when every score shifts by a constant
is broken in a way nothing downstream would notice.
"""

from __future__ import annotations

import chess
import pytest
import torch

from rukh.models.config import EncoderConfig
from rukh.models.encoder import PositionEncoder
from rukh.models.reward import RewardModel, preference_accuracy, preference_loss
from rukh.models.squares import SQUARE_TOKENS, fen_to_tokens
from rukh.train.reward import RewardExample, encode_pairs, split_examples

pytestmark = pytest.mark.unit

TOY = EncoderConfig(input="squares", n_layer=1, n_head=2, d_model=16, dropout=0.0)


def _model() -> RewardModel:
    torch.manual_seed(0)
    return RewardModel(PositionEncoder(TOY))


def test_the_loss_only_sees_the_difference():
    """Shift every score by a constant and nothing changes: that *is* Bradley-Terry."""
    chosen = torch.tensor([1.0, -0.5, 3.0])
    rejected = torch.tensor([0.0, -2.0, 1.0])
    base = preference_loss(chosen, rejected)
    for shift in (-100.0, 0.25, 7.0):
        assert preference_loss(chosen + shift, rejected + shift) == pytest.approx(float(base))


def test_a_perfect_ordering_drives_the_loss_towards_zero():
    wide = torch.tensor([20.0, 20.0])
    low = torch.tensor([-20.0, -20.0])
    assert float(preference_loss(wide, low)) == pytest.approx(0.0, abs=1e-6)
    assert preference_accuracy(wide, low) == 1.0
    # And a coin flip costs log 2 per pair, which is the loss of a model that knows nothing.
    same = torch.zeros(4)
    assert float(preference_loss(same, same)) == pytest.approx(0.6931, abs=1e-3)


def test_a_tie_counts_as_wrong_rather_than_as_half_right():
    """A model that scores two moves the same has expressed no preference."""
    same = torch.tensor([1.0, 1.0])
    assert preference_accuracy(same, same) == 0.0


def test_the_model_reads_a_position_and_answers_one_number_per_row():
    model = _model()
    board = chess.Board()
    board.push_uci("e2e4")
    idx = torch.tensor([fen_to_tokens(board.fen())] * 3, dtype=torch.long)
    assert idx.shape == (3, SQUARE_TOKENS)
    scores = model(idx)
    assert scores.shape == (3,)
    # The same position scores the same, which sounds obvious and is the pooling being right.
    assert float(scores.detach().std()) == pytest.approx(0.0, abs=1e-6)


def test_a_row_that_does_not_replay_is_dropped_rather_than_repaired():
    """Guessing which of the dataset and python-chess is right is how bad positions creep in."""
    rows = [
        {
            "game_id": "1",
            "prefix": "e2e4",
            "chosen": "e7e5",
            "rejected": "h7h5",
            "cp_chosen": 10,
            "cp_rejected": 110,
        },
        # An illegal "chosen" for this prefix: white cannot move twice.
        {
            "game_id": "2",
            "prefix": "e2e4",
            "chosen": "d2d4",
            "rejected": "e7e5",
            "cp_chosen": 10,
            "cp_rejected": 110,
        },
        # A prefix that is not a legal game at all.
        {
            "game_id": "3",
            "prefix": "e2e9",
            "chosen": "e7e5",
            "rejected": "h7h5",
            "cp_chosen": 10,
            "cp_rejected": 110,
        },
    ]
    kept = encode_pairs(rows)
    assert [example.game_id for example in kept] == ["1"]
    assert len(kept[0].chosen) == len(kept[0].rejected) == SQUARE_TOKENS
    assert kept[0].delta_cp == 100


def test_the_split_goes_by_game_so_two_pairs_of_one_game_cannot_straddle_it():
    """Two pairs a few plies apart are almost the same pair; splitting by pair leaks."""
    examples = [
        RewardExample(
            game_id=str(game),
            chosen=[0] * 4,
            rejected=[0] * 4,
            white_moved=True,
            delta_cp=100.0,
        )
        for game in range(200)
        for _ in range(3)
    ]
    train, val = split_examples(examples, val_fraction=0.2, seed=42)
    assert train and val
    assert len(train) + len(val) == len(examples)
    overlap = {e.game_id for e in train} & {e.game_id for e in val}
    assert overlap == set(), "no game may appear on both sides of the split"
    # Every game keeps all three of its pairs on the same side.
    for side in (train, val):
        counts = {}
        for example in side:
            counts[example.game_id] = counts.get(example.game_id, 0) + 1
        assert set(counts.values()) == {3}


def test_training_one_batch_moves_the_scores_apart():
    """The smallest end-to-end check: the loss goes down and the ordering improves."""
    model = _model()
    board = chess.Board()
    board.push_uci("e2e4")
    good = torch.tensor([fen_to_tokens(board.fen())], dtype=torch.long)
    board.pop()
    board.push_uci("f2f3")
    bad = torch.tensor([fen_to_tokens(board.fen())], dtype=torch.long)

    optimiser = torch.optim.AdamW(model.parameters(), lr=0.05)
    first = float(preference_loss(model(good), model(bad)))
    for _ in range(30):
        loss = preference_loss(model(good), model(bad))
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
    last = float(preference_loss(model(good), model(bad)))
    assert last < first
    assert preference_accuracy(model(good), model(bad)) == 1.0
