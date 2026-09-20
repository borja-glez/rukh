"""On-policy pairs: the parts that decide whether the comparison with off-policy is fair.

Two datasets are only comparable when the one thing that differs between them is the thing
under study. Here that thing is *where the two moves came from*, so the tests are about
everything else: the positions, the scoring convention and the prompt.
"""

from __future__ import annotations

import chess
import pytest

from rukh.data.onpolicy import OnPolicyConfig, sample_candidates, score_move
from rukh.data.scoring import MATE_SCORE

pytestmark = pytest.mark.unit


class _FakeInfo(dict):
    pass


class _FakeEngine:
    """An engine that answers with a fixed score from White's point of view.

    ``python-chess`` hands back a ``PovScore``, and getting its point of view wrong is a bug that
    does not crash: it silently builds a dataset whose ``chosen`` is the worse move.
    """

    def __init__(self, cp_for_white: int) -> None:
        self.cp_for_white = cp_for_white
        self.asked: list[str] = []

    def analyse(self, board: chess.Board, limit: object) -> dict[str, object]:
        self.asked.append(board.fen())
        return {"score": chess.engine.PovScore(chess.engine.Cp(self.cp_for_white), chess.WHITE)}


def test_the_score_comes_back_for_the_side_that_played_the_move():
    """+120 for White is +120 when White moved and -120 when Black did."""
    engine = _FakeEngine(120)
    board = chess.Board()
    assert score_move(engine, board, chess.Move.from_uci("e2e4"), depth=4) == 120

    board.push_uci("e2e4")
    assert score_move(engine, board, chess.Move.from_uci("e7e5"), depth=4) == -120


def test_scoring_a_move_leaves_the_board_exactly_as_it_found_it():
    """The loop reuses one board for every candidate; a stray push poisons the rest."""
    engine = _FakeEngine(0)
    board = chess.Board()
    board.push_uci("d2d4")
    before = board.fen()
    score_move(engine, board, chess.Move.from_uci("d7d5"), depth=4)
    assert board.fen() == before


@pytest.mark.parametrize(("moves", "expected"), [(1, MATE_SCORE - 1), (5, MATE_SCORE - 5)])
def test_a_mate_is_scored_with_the_project_convention(moves: int, expected: int):
    """``MATE_SCORE`` minus the distance, the same convention the off-policy column uses.

    The subtraction is not decoration: it is what makes a mate in one preferable to a mate in
    five, which is a preference a pair should be able to express.
    """

    class _Mate(_FakeEngine):
        def analyse(self, board: chess.Board, limit: object) -> dict[str, object]:
            return {"score": chess.engine.PovScore(chess.engine.Mate(moves), chess.WHITE)}

    assert score_move(_Mate(0), chess.Board(), chess.Move.from_uci("e2e4"), depth=4) == expected


class _Tokenizer:
    """Just enough of ``UciTokenizer`` for the sampler stub below."""

    unk_id = 0
    vocab: dict[str, int] = {}


def test_only_distinct_legal_moves_survive_sampling(monkeypatch):
    """An illegal proposal is dropped, not rescued, and a repeat is not a second candidate."""
    proposals = iter(
        [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e2e4"),  # a repeat: the common case at any temperature
            chess.Move.from_uci("e7e5"),  # illegal for White
            None,  # the sampler can decline
            chess.Move.from_uci("d2d4"),
        ]
    )

    import rukh.infer.sampler as sampler

    monkeypatch.setattr(sampler, "model_generator", lambda model, cfg: None)
    monkeypatch.setattr(sampler, "pick_move", lambda *args, **kwargs: (next(proposals, None), True))

    moves = sample_candidates(
        model=None,  # type: ignore[arg-type]
        tok=_Tokenizer(),  # type: ignore[arg-type]
        board=chess.Board(),
        history=[],
        count=4,
        temperature=1.0,
        top_k=20,
    )
    assert [move.uci() for move in moves] == ["e2e4", "d2d4"]


def test_the_config_refuses_a_setting_that_cannot_make_a_pair():
    """One candidate is not a preference, and temperature zero is not a distribution."""
    with pytest.raises(ValueError):
        OnPolicyConfig(candidates=1)
    with pytest.raises(ValueError):
        OnPolicyConfig(temperature=0.0)
    with pytest.raises(ValueError):
        OnPolicyConfig(depth=0)


def test_the_prompt_is_the_one_every_other_measurement_uses():
    """The header tokens come from ``infer.game``, so a pair is built from the same prompt the
    ladder and the demo read. Hand-rolling them here is how two measurements quietly diverge."""
    from rukh.data.onpolicy import _history_for
    from rukh.infer.game import _history
    from rukh.tokenize.uci_vocab import UciTokenizer

    tok = UciTokenizer()
    plain = _history(tok, 1600, 1600)
    with_moves = _history_for(tok, "e2e4 e7e5", 1600, 1600)
    assert with_moves[: len(plain)] == plain
    assert with_moves[len(plain) :] == [tok.vocab["e2e4"], tok.vocab["e7e5"]]
