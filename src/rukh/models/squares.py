"""The ``squares`` input scheme: a FEN as 69 fixed tokens, the didactic rival of ``moves``.

The ``moves`` scheme feeds the encoder the game so far (the decoder's own UCI vocabulary); this
one feeds it the position itself, so the two can be compared on the same heads. That comparison
is the point of the module: it is the "what is a good representation?" lesson of M3.

Layout (always exactly ``SQUARE_TOKENS`` = 69 positions, so no padding is ever needed)::

    0        <cls>                 pooling anchor, the counterpart of the decoder's <bos>
    1..64    the 64 squares        piece or <empty>, file-major (a1, a2, ..., a8, b1, ..., h8)
    65       side to move          turn:w / turn:b
    66       castling rights       castle:<KQkq subset>, 16 combinations in one token
    67       en passant            ep:none or the file, ep:a ... ep:h
    68       halfmove clock        clock:0 ... clock:3, the bucketed 50-move counter

The square order is ``rukh.tokenize.uci_vocab.squares()``, file-major, the same enumeration the
UCI vocabulary uses for move endpoints: one ordering for the whole project.

The plan's arithmetic (64 + turn + 4 castling + en passant + clock) adds up to 71, not to the 69
it fixes; 69 is the binding number, so the four castling rights travel in one token with 16
values (they are four bits of one fact, and this keeps every other component in a slot of its
own) and the position that frees up became ``<cls>``, which gives ``pool("cls")`` a real anchor
in this scheme instead of borrowing square a1.

Halfmove-clock buckets, chosen so that the boundaries mean something over the board:

===========  ===================================================================
``clock:0``  0-5 half-moves: a pawn moved or a piece was captured very recently
``clock:1``  6-24: a normal manoeuvring stretch
``clock:2``  25-49: the 50-move rule is in sight and shapes the plan
``clock:3``  50 or more: a draw can be claimed
===========  ===================================================================

A four-field FEN (``fen4``, the key of the P1 positions and evaluations) carries no counters, so
it reads back as ``clock:0``; that is the value every label built from P1 data will have.

``SQUARE_VOCAB`` is fixed by construction, never learned from data, and ``vocab_hash`` pins it:
a change to the enumeration invalidates every encoder trained with it, exactly like the UCI
vocabulary of P1.
"""

from __future__ import annotations

import hashlib
import json

from rukh.tokenize.uci_vocab import FILES, squares

PIECES = "PNBRQKpnbrqk"
CASTLING_ORDER = "KQkq"
N_SQUARES = 64
SQUARE_TOKENS = 69
"""Length of every ``squares`` sequence: 1 + 64 + 4."""

CLOCK_EDGES = (6, 25, 50)
"""Upper edges (exclusive) of the halfmove-clock buckets; see the module docstring."""


def castling_strings() -> list[str]:
    """The 16 castling combinations in a fixed order: ``-``, ``K``, ``Q``, ``KQ``, ``k``, ..."""
    out: list[str] = []
    for mask in range(16):
        rights = "".join(right for bit, right in enumerate(CASTLING_ORDER) if mask >> bit & 1)
        out.append(rights or "-")
    return out


def build_square_vocab() -> list[str]:
    """The token list in id order; see the module docstring for the layout it serves."""
    tokens = ["<pad>", "<mask>", "<cls>", "<empty>"]
    tokens.extend(PIECES)
    tokens.extend(["turn:w", "turn:b"])
    tokens.extend(f"castle:{rights}" for rights in castling_strings())
    tokens.append("ep:none")
    tokens.extend(f"ep:{file}" for file in FILES)
    tokens.extend(f"clock:{index}" for index in range(len(CLOCK_EDGES) + 1))
    return tokens


SQUARE_VOCAB: list[str] = build_square_vocab()
SQUARE_IDS: dict[str, int] = {token: index for index, token in enumerate(SQUARE_VOCAB)}
SQUARE_VOCAB_SIZE = len(SQUARE_VOCAB)

PAD_ID = SQUARE_IDS["<pad>"]
MASK_ID = SQUARE_IDS["<mask>"]
CLS_ID = SQUARE_IDS["<cls>"]
EMPTY_ID = SQUARE_IDS["<empty>"]
CONTROL_IDS: frozenset[int] = frozenset({PAD_ID, MASK_ID, CLS_ID})
"""Never masked by ``apply_masking``: they are the frame of the sequence, not a prediction."""


def vocab_hash() -> str:
    """SHA-256 of the vocabulary as a JSON list: the identity of the ``squares`` scheme."""
    payload = json.dumps(SQUARE_VOCAB, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def clock_bucket(halfmove: int) -> int:
    """Index of the halfmove-clock bucket of ``halfmove`` (see ``CLOCK_EDGES``)."""
    if halfmove < 0:
        raise ValueError(f"the halfmove clock cannot be negative, got {halfmove}")
    return sum(halfmove >= edge for edge in CLOCK_EDGES)


def _placement_ids(placement: str) -> list[int]:
    """The 64 square ids, file-major, from the piece-placement field of a FEN."""
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"a FEN placement needs 8 ranks, got {len(ranks)}: {placement!r}")
    ids = [EMPTY_ID] * N_SQUARES
    for row, rank_text in enumerate(ranks):
        rank = 7 - row  # the placement is written from rank 8 down to rank 1
        file = 0
        for char in rank_text:
            if char.isdigit():
                file += int(char)
            elif char in PIECES:
                if file >= 8:
                    raise ValueError(f"rank {rank + 1} of {placement!r} is too long")
                ids[file * 8 + rank] = SQUARE_IDS[char]
                file += 1
            else:
                raise ValueError(f"unknown piece {char!r} in {placement!r}")
        if file != 8:
            raise ValueError(f"rank {rank + 1} of {placement!r} covers {file} files, not 8")
    return ids


def _castling_id(field: str) -> int:
    """Token id of a FEN castling field, normalized to the ``KQkq`` order."""
    if field in ("-", ""):
        return SQUARE_IDS["castle:-"]
    unknown = [char for char in field if char not in CASTLING_ORDER]
    if unknown:
        raise ValueError(f"unsupported castling field {field!r} (Chess960 is out of scope)")
    rights = "".join(right for right in CASTLING_ORDER if right in field)
    return SQUARE_IDS[f"castle:{rights}"]


def fen_to_tokens(fen: str) -> list[int]:
    """The ``SQUARE_TOKENS`` ids of a FEN; four fields (``fen4``) or the full six are accepted.

    Raises ``ValueError`` on a malformed FEN: the encoder must never be fed a position that was
    silently repaired into a different one.
    """
    fields = fen.split()
    if len(fields) < 4:
        raise ValueError(f"a FEN needs at least 4 fields, got {len(fields)}: {fen!r}")
    placement, turn, castling, ep = fields[:4]
    if turn not in ("w", "b"):
        raise ValueError(f"the side to move must be 'w' or 'b', got {turn!r}")
    halfmove = int(fields[4]) if len(fields) > 4 else 0
    if ep in ("-", ""):
        ep_token = "ep:none"
    elif len(ep) == 2 and ep[0] in FILES and ep[1] in "36":
        ep_token = f"ep:{ep[0]}"
    else:
        raise ValueError(f"unknown en-passant square {ep!r}")
    return [
        CLS_ID,
        *_placement_ids(placement),
        SQUARE_IDS[f"turn:{turn}"],
        _castling_id(castling),
        SQUARE_IDS[ep_token],
        SQUARE_IDS[f"clock:{clock_bucket(halfmove)}"],
    ]


def tokens_to_strings(tokens: list[int]) -> list[str]:
    """The token strings of a sequence, for debugging and for the lesson's figures."""
    return [SQUARE_VOCAB[token] for token in tokens]


def tokens_to_fen(tokens: list[int]) -> str:
    """The four-field FEN a token sequence encodes: the inverse of ``fen_to_tokens``.

    The halfmove-clock bucket is dropped because it is lossy by construction, so the round trip
    is exact for a ``fen4`` and exact up to the counters for a full FEN.
    """
    if len(tokens) != SQUARE_TOKENS:
        raise ValueError(f"a squares sequence has {SQUARE_TOKENS} tokens, got {len(tokens)}")
    names = tokens_to_strings(tokens)
    if names[0] != "<cls>":
        raise ValueError(f"a squares sequence starts with <cls>, got {names[0]!r}")
    board = names[1 : 1 + N_SQUARES]
    ranks: list[str] = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            piece = board[file * 8 + rank]
            if piece == "<empty>":
                empty += 1
                continue
            if piece not in PIECES:
                raise ValueError(f"{piece!r} is not a piece token")
            row += (str(empty) if empty else "") + piece
            empty = 0
        ranks.append(row + (str(empty) if empty else ""))
    turn = names[1 + N_SQUARES].removeprefix("turn:")
    castling = names[2 + N_SQUARES].removeprefix("castle:")
    ep_file = names[3 + N_SQUARES].removeprefix("ep:")
    ep = "-" if ep_file == "none" else f"{ep_file}{'6' if turn == 'w' else '3'}"
    return f"{'/'.join(ranks)} {turn} {castling} {ep}"


def square_name(index: int) -> str:
    """Name of the board square at position ``index`` of the 64-square block (0 = ``a1``)."""
    return squares()[index]
