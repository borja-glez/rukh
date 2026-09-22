"""Export one game's encoder and Stockfish evaluations to artifacts/web/value-bar.json."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.engine
import torch

from rukh.engine import find_stockfish
from rukh.models.squares import fen_to_tokens
from rukh.train import load_any
from rukh.train.checkpoint import resolve_run

# `encoder-heads.yaml` sets `unique_run_name: true`, so on disk the run is stamped
# (`encoder-heads-20260921-185425/`) and the unstamped folder never exists. `resolve_run` takes
# the newest run of that name, which is what the config means by `run_name: encoder-heads`.
CKPT = resolve_run(Path("checkpoints/encoder-heads/best.pt"))
OUT = Path("artifacts/web/value-bar.json")
VALUE_SCALE = 400.0  # the tanh(cp / 400) of rukh.data.labels: both curves must share it
THRESHOLD = 0.5  # the blunder threshold of configs/eval/encoder.yaml
DEPTH = 12  # Stockfish depth; deeper is slower and barely moves the curve at club level
# Byrne-Fischer, New York 1956: the Game of the Century, where 17...Be6 offers the queen.
MOVES = (
    "g1f3 g8f6 c2c4 g7g6 b1c3 f8g7 d2d4 e8g8 c1f4 d7d5 d1b3 d5c4 b3c4 c7c6 e2e4 b8d7 "
    "a1d1 d7b6 c4c5 c8g4 f4g5 b6a4 c5a3 a4c3 b2c3 f6e4 g5e7 d8b6 f1c4 e4c3 e7c5 f8e8 "
    "e1f1 g4e6"
).split()

model, _payload, kind = load_any(CKPT)
if kind != "encoder":
    raise SystemExit(f"{CKPT} is a {kind} checkpoint; the value bar needs the encoder heads")
model.eval()

engine_path = find_stockfish()
if engine_path is None:
    raise SystemExit("Stockfish not found: the second curve is the whole point of this figure")

board = chess.Board()
san: list[str] = []
fens: list[str] = []
for uci in MOVES:  # fail loudly if the line is not legal
    move = chess.Move.from_uci(uci)
    san.append(board.san(move))
    board.push(move)
    fens.append(board.fen())

idx = torch.tensor([fen_to_tokens(fen) for fen in fens], dtype=torch.long)
with torch.no_grad():
    outputs = model(idx)
values = outputs["value"].float().tolist()
blunders = torch.sigmoid(outputs["blunder"].float()).tolist()

stockfish: list[float | None] = []
with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
    for fen in fens:
        info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=DEPTH))
        score = info["score"].white()
        cp = score.score(mate_score=10000)
        stockfish.append(None if cp is None else math.tanh(cp / VALUE_SCALE))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(
        {
            "schema": "rukh-value-bar/1",
            "game": {"moves": MOVES, "san": san},
            "series": [
                {
                    "ply": ply + 1,
                    "encoder": round(values[ply], 4),
                    "stockfish": None if stockfish[ply] is None else round(stockfish[ply], 4),
                    "blunder": blunders[ply] > THRESHOLD,
                }
                for ply in range(len(MOVES))
            ],
            "meta": {
                "model": "rukh-encoder",
                "checkpoint": str(CKPT),
                "white": "Byrne, D.",
                "black": "Fischer, R.",
                "event": "New York, 1956",
                "result": "0-1",
                "value_scale": VALUE_SCALE,
                "threshold": THRESHOLD,
                "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        }
    ),
    encoding="utf-8",
)
print(f"wrote {OUT} ({len(MOVES)} plies, depth {DEPTH})")
