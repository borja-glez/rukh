"""Fixed UCI vocabulary: a deterministic enumeration shared with the TypeScript twin.

The vocabulary does not depend on data. It is built by enumerating squares in file-major order
(``a1, a2, ..., a8, b1, ..., h8``), every ``from``/``to`` pair reachable by a queen or a knight
(1 792 moves), every promotion (176) and a fixed block of special tokens in front. Any change to
this enumeration breaks parity with ``rukh-web/src/lib/chess-lm/tokenizer.ts`` and with
``artifacts/tokenizer/fixtures/games.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

FILES = "abcdefgh"
PROMOTION_PIECES = "qrbn"
RESULT_TOKENS: dict[str, str] = {"1-0": "<1-0>", "0-1": "<0-1>", "1/2-1/2": "<1/2>"}
ELO_MIN = 600
ELO_MAX = 3200
ELO_STEP = 100

SPECIALS: list[str] = ["<pad>", "<bos>", "<eos>", "<mask>", "<unk>", "<1-0>", "<0-1>", "<1/2>"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3
UNK_ID = 4


def squares() -> list[str]:
    """The 64 squares in file-major order (index ``file * 8 + rank``)."""
    return [f"{FILES[f]}{r + 1}" for f in range(8) for r in range(8)]


def _reachable(df: int, dr: int) -> bool:
    """Whether a queen (line or diagonal) or a knight can go from one square to another."""
    if df == 0 and dr == 0:
        return False
    if df == 0 or dr == 0 or abs(df) == abs(dr):
        return True
    return {abs(df), abs(dr)} == {1, 2}


def build_moves() -> list[str]:
    """1 792 ``fromto`` strings, ``from`` outer and ``to`` inner, both in file-major order."""
    moves: list[str] = []
    for ff in range(8):
        for fr in range(8):
            for tf in range(8):
                for tr in range(8):
                    if _reachable(tf - ff, tr - fr):
                        moves.append(f"{FILES[ff]}{fr + 1}{FILES[tf]}{tr + 1}")
    return moves


def build_promotions() -> list[str]:
    """176 ``fromto+piece`` strings: rank 7 to 8 (White) then rank 2 to 1 (Black)."""
    promotions: list[str] = []
    for from_rank, to_rank in ((7, 8), (2, 1)):
        for ff in range(8):
            for tf in range(8):
                if abs(tf - ff) <= 1:
                    for piece in PROMOTION_PIECES:
                        promotions.append(f"{FILES[ff]}{from_rank}{FILES[tf]}{to_rank}{piece}")
    return promotions


def elo_tokens(side: str) -> list[str]:
    """``<w0600>`` ... ``<w3200>`` (or ``b``): 27 bins of 100 Elo."""
    return [f"<{side}{elo:04d}>" for elo in range(ELO_MIN, ELO_MAX + 1, ELO_STEP)]


def build_vocab() -> list[str]:
    """The full token list in id order: specials, Elo bins, moves, promotions (2 030 tokens)."""
    return SPECIALS + elo_tokens("w") + elo_tokens("b") + build_moves() + build_promotions()


def elo_bin(elo: int) -> int:
    """Lower edge of the 100-Elo bin, clamped to ``[600, 3200]``."""
    return min(max(int(elo), ELO_MIN), ELO_MAX + ELO_STEP - 1) // ELO_STEP * ELO_STEP


def elo_token(elo: int, side: str) -> str:
    """``<w1800>``-style token for ``elo`` and side ``w`` or ``b``."""
    if side not in ("w", "b"):
        raise ValueError(f"side must be 'w' or 'b', got {side!r}")
    return f"<{side}{elo_bin(elo):04d}>"


class UciTokenizer:
    """Encode/decode games as ``[<bos>, <wXXXX>, <bXXXX>, moves..., <result>, <eos>]``."""

    specials: list[str] = SPECIALS
    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID
    mask_id = MASK_ID
    unk_id = UNK_ID

    def __init__(self) -> None:
        self.ids: list[str] = build_vocab()
        self.vocab: dict[str, int] = {token: i for i, token in enumerate(self.ids)}
        if len(self.vocab) != len(self.ids):
            raise RuntimeError("duplicate tokens in the UCI vocabulary")

    def __len__(self) -> int:
        return len(self.ids)

    def encode_game(
        self,
        uci: str,
        white_elo: int,
        black_elo: int,
        result: str,
        max_len: int = 200,
    ) -> list[int]:
        """Encode one game; unknown moves map to ``<unk>``; the sequence is cut at ``max_len``.

        When the whole game fits in ``max_len`` the sequence ends with ``<eos>``; otherwise it
        is truncated to the first ``max_len`` ids (no result, no ``<eos>``).
        """
        if result not in RESULT_TOKENS:
            raise ValueError(f"unknown result {result!r}; expected one of {list(RESULT_TOKENS)}")
        ids = [
            self.bos_id,
            self.vocab[elo_token(white_elo, "w")],
            self.vocab[elo_token(black_elo, "b")],
        ]
        ids.extend(self.vocab.get(move, self.unk_id) for move in uci.split())
        ids.append(self.vocab[RESULT_TOKENS[result]])
        ids.append(self.eos_id)
        return ids[:max_len]

    def decode(self, ids: list[int]) -> list[str]:
        """Map ids back to their token strings (raises ``IndexError`` on unknown ids)."""
        return [self.ids[i] for i in ids]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "size": len(self.ids),
            "specials": list(self.specials),
            "elo_bins": {"min": ELO_MIN, "max": ELO_MAX, "step": ELO_STEP},
            "tokens": list(self.ids),
        }

    def export(self, path: Path) -> None:
        """Write ``vocab.json`` deterministically (``indent=0``, ASCII, trailing newline)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=0, ensure_ascii=True) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")

    @classmethod
    def from_file(cls, path: Path) -> UciTokenizer:
        """Load a ``vocab.json`` and check it matches the built-in enumeration."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        if data.get("tokens") != tokenizer.ids:
            raise ValueError(f"{path} does not match the built-in UCI vocabulary")
        return tokenizer
