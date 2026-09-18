"""Inference: masked sampling of one move and self-play against an opponent."""

from rukh.infer.game import (
    GameResult,
    Opponent,
    RandomOpponent,
    StockfishOpponent,
    legality_rate,
    play_game,
    result_token,
)
from rukh.infer.sampler import SampleConfig, legal_token_ids, pick_move

__all__ = [
    "GameResult",
    "Opponent",
    "RandomOpponent",
    "SampleConfig",
    "StockfishOpponent",
    "legal_token_ids",
    "legality_rate",
    "pick_move",
    "play_game",
    "result_token",
]
