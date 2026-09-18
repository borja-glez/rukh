"""Char-level SAN tokenizer: a fixed 32-symbol alphabet plus ``<pad>``, ``<bos>``, ``<eos>``.

The text is the numbered SAN movetext without comments followed by the result:
``1.e4 e5 2.Nf3 Nc6 ... 1-0``. Every character must belong to the alphabet.
"""

from __future__ import annotations

ALPHABET = " #+-.0123456789=BKNOQRabcdefghx/"
SPECIALS: list[str] = ["<pad>", "<bos>", "<eos>"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class SanCharTokenizer:
    """Encode text as ``[<bos>, char..., <eos>]`` and decode ids back to text."""

    specials: list[str] = SPECIALS
    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID

    def __init__(self) -> None:
        self.ids: list[str] = SPECIALS + list(ALPHABET)
        self.vocab: dict[str, int] = {token: i for i, token in enumerate(self.ids)}
        if len(self.vocab) != len(self.ids):
            raise RuntimeError("duplicate symbols in the SAN alphabet")

    def __len__(self) -> int:
        return len(self.ids)

    def encode(self, text: str) -> list[int]:
        """Encode ``text``; raises ``ValueError`` on a character outside the alphabet."""
        ids = [self.bos_id]
        for ch in text:
            try:
                ids.append(self.vocab[ch])
            except KeyError:
                raise ValueError(f"character {ch!r} is not in the SAN alphabet") from None
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Join the characters of ``ids``, skipping special tokens."""
        return "".join(self.ids[i] for i in ids if i >= len(SPECIALS))


def san_text(san_movetext: str, result: str) -> str:
    """The exact text this scheme encodes: numbered SAN plus a space and the result."""
    return f"{san_movetext} {result}"


def uci_to_san(uci: str) -> str:
    """Numbered SAN movetext (``1.e4 e5 2.Nf3 ...``) for a space-separated UCI game.

    Raises ``ValueError`` on an illegal move, like the rest of the pipeline.
    """
    import chess

    board = chess.Board()
    out: list[str] = []
    for token in uci.split():
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {token} in {board.fen()}")
        san = board.san(move)
        out.append(f"{board.fullmove_number}.{san}" if board.turn == chess.WHITE else san)
        board.push(move)
    return " ".join(out)
