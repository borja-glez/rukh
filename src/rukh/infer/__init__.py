"""Inference: masked sampling of one move and self-play against an opponent."""

from rukh.infer.game import (
    ADJUDICATION_CP,
    ADJUDICATION_DEPTH,
    GameResult,
    Opponent,
    RandomOpponent,
    StockfishOpponent,
    adjudicate,
    is_over,
    legality_rate,
    material_balance,
    play_game,
    result_token,
)
from rukh.infer.sampler import (
    HEADER_TOKENS,
    SampleConfig,
    legal_token_ids,
    model_generator,
    pick_move,
    prompt_ids,
)

__all__ = [
    "ADJUDICATION_CP",
    "ADJUDICATION_DEPTH",
    "HEADER_TOKENS",
    "GameResult",
    "Opponent",
    "RandomOpponent",
    "SampleConfig",
    "StockfishOpponent",
    "adjudicate",
    "is_over",
    "legal_token_ids",
    "legality_rate",
    "material_balance",
    "model_generator",
    "pick_move",
    "play_game",
    "prompt_ids",
    "result_token",
]
