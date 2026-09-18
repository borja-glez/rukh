"""BPE over UCI text with Hugging Face ``tokenizers``.

The text of one game is its UCI moves concatenated without spaces (``e2e4e7e5g1f3...``), so
that BPE can merge whole openings into one token. ``WhitespaceSplit`` keeps games apart when
several are fed as one text. The model has no normalizer, no continuing-subword prefix and no
end-of-word suffix: the TypeScript twin re-applies ``model.merges`` by rank over each
whitespace-split word and must reproduce ``Tokenizer.encode(text).ids`` exactly.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from rukh.tokenize.uci_vocab import UciTokenizer

if TYPE_CHECKING:
    import chess

PROMOTION_LETTERS = frozenset("qrbn")


def bpe_text(uci: str) -> str:
    """The text BPE sees for one game: moves joined without spaces."""
    return uci.replace(" ", "")


def split_moves(text: str) -> list[str]:
    """Split a spaceless UCI string back into moves.

    A move is four characters plus a promotion letter, and ``b`` is both a file and a
    promotion piece, so the split replays the game: the letter belongs to the move only when
    the piece that moves is a pawn reaching the last rank (or, when the position is corrupt
    and no piece sits on the origin square, when the ranks are 7 to 8 or 2 to 1).
    """
    import chess

    board = chess.Board()
    moves: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        move = text[i : i + 4]
        i += 4
        if i < n and text[i] in PROMOTION_LETTERS and _is_promotion(board, move):
            move += text[i]
            i += 1
        moves.append(move)
        try:
            parsed = chess.Move.from_uci(move)
        except ValueError:
            break
        if parsed in board.legal_moves:
            board.push(parsed)
        else:
            break
    if i < n:
        moves.extend(_split_blind(text[i:]))
    return moves


def _is_promotion(board: chess.Board, move: str) -> bool:
    import chess

    if len(move) != 4:
        return False
    try:
        from_sq = chess.parse_square(move[:2])
        to_sq = chess.parse_square(move[2:])
    except ValueError:
        return False
    piece = board.piece_at(from_sq)
    if piece is None:
        return move[1] + move[3] in ("78", "21") and abs(ord(move[0]) - ord(move[2])) <= 1
    return piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7)


def _split_blind(text: str) -> list[str]:
    """Rank-based split for text after an illegal move (no board to consult)."""
    moves: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        move = text[i : i + 4]
        i += 4
        if (
            i < n
            and text[i] in PROMOTION_LETTERS
            and len(move) == 4
            and move[1] + move[3] in ("78", "21")
            and abs(ord(move[0]) - ord(move[2])) <= 1
        ):
            move += text[i]
            i += 1
        moves.append(move)
    return moves


def train_bpe(
    texts: Iterable[str],
    vocab_size: int = 4096,
    specials: list[str] | None = None,
) -> Tokenizer:
    """Train a BPE model on ``texts`` (one game per text, already spaceless)."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(UciTokenizer.specials if specials is None else specials),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def save(tokenizer: Tokenizer, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path), pretty=True)


def load_bpe(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text).ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    """Concatenate the tokens (skipping specials) and re-split into space-separated moves."""
    specials = set(UciTokenizer.specials)
    tokens = [tokenizer.id_to_token(i) for i in ids]
    text = "".join(t for t in tokens if t is not None and t not in specials)
    return " ".join(split_moves(text))


def iter_parquet_texts(games: Path, n_games: int) -> Iterator[str]:
    """Yield ``bpe_text`` for the first ``n_games`` rows of a UCI games parquet."""
    import polars as pl

    frame = pl.scan_parquet(Path(games).as_posix()).select("uci").head(n_games).collect()
    for uci in frame["uci"]:
        yield bpe_text(uci)


def train_bpe_from_parquet(games: Path, out: Path, vocab_size: int, n_games: int) -> Tokenizer:
    tokenizer = train_bpe(iter_parquet_texts(games, n_games), vocab_size=vocab_size)
    save(tokenizer, out)
    return tokenizer


def longest_tokens(tokenizer: Tokenizer, n: int = 10) -> list[dict[str, object]]:
    """The ``n`` longest learned tokens, with how many whole moves each spans."""
    specials = set(UciTokenizer.specials)
    vocab = tokenizer.get_vocab()
    tokens = sorted((t for t in vocab if t not in specials), key=lambda t: (-len(t), vocab[t]))[:n]
    return [{"token": t, "id": vocab[t], "length": len(t), "moves": len(t) // 4} for t in tokens]
