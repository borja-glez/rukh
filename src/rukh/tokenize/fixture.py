"""Parity fixture: 20 PGN games encoded with every scheme, consumed by the TypeScript twin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import chess.pgn

from rukh.tokenize.uci_vocab import UciTokenizer

if TYPE_CHECKING:
    from tokenizers import Tokenizer

    from rukh.tokenize.san_chars import SanCharTokenizer


def read_pgn_games(path: Path) -> list[dict[str, object]]:
    """Read every game of a PGN file as ``{id, white_elo, black_elo, result, uci, san}``.

    ``san`` is the numbered movetext without comments or result (``1.e4 e5 2.Nf3 ...``) and
    ``uci`` the space-separated UCI moves. Every move is replayed with python-chess, so an
    illegal game raises instead of producing a corrupt fixture.
    """
    games: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            if game.errors:
                raise ValueError(f"{path}: {game.headers.get('Site')}: {game.errors[0]}")
            board = game.board()
            uci: list[str] = []
            san: list[str] = []
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    raise ValueError(f"{path}: illegal move {move.uci()} in {board.fen()}")
                if board.turn == chess.WHITE:
                    san.append(f"{board.fullmove_number}.{board.san(move)}")
                else:
                    san.append(board.san(move))
                uci.append(move.uci())
                board.push(move)
            games.append(
                {
                    "id": game.headers.get("Site", f"game-{len(games) + 1}"),
                    "white_elo": int(game.headers["WhiteElo"]),
                    "black_elo": int(game.headers["BlackElo"]),
                    "result": game.headers["Result"],
                    "uci": " ".join(uci),
                    "san": " ".join(san),
                }
            )
    return games


def build_fixture(
    pgn_path: Path,
    uci_tokenizer: UciTokenizer,
    san_tokenizer: SanCharTokenizer | None = None,
    bpe: Tokenizer | None = None,
    max_len: int = 200,
) -> list[dict[str, object]]:
    """Encode the PGN games with the fixed vocabulary and, when given, char-level SAN and BPE."""
    entries = read_pgn_games(pgn_path)
    for entry in entries:
        entry["uci_ids"] = uci_tokenizer.encode_game(
            str(entry["uci"]),
            int(entry["white_elo"]),  # type: ignore[arg-type]
            int(entry["black_elo"]),  # type: ignore[arg-type]
            str(entry["result"]),
            max_len=max_len,
        )
        if san_tokenizer is not None:
            entry["san_ids"] = san_tokenizer.encode(f"{entry['san']} {entry['result']}")
        if bpe is not None:
            entry["bpe_ids"] = bpe.encode(str(entry["uci"])).ids
    return entries


def write_fixture(entries: list[dict[str, object]], path: Path) -> None:
    """Write the fixture deterministically (``indent=0``, ASCII, trailing newline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(entries, indent=0, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
