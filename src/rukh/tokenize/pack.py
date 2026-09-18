"""Pack encoded games into one ``uint16`` stream (``tokens.npy``) plus a game index.

The stream is the concatenation of every encoded game (``<bos> ... <eos>``), untruncated;
``starts.npy`` holds the offset of each ``<bos>`` so a loader can cut ``block``-sized windows
aligned to game starts (D-019). ``meta.json`` records counts, the scheme and a hash of the
vocabulary so a stale pack is never mixed with a different tokenizer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from rukh.data.pipeline import TokenizeConfig
from rukh.paths import resolve
from rukh.tokenize.san_chars import SanCharTokenizer, san_text, uci_to_san
from rukh.tokenize.uci_vocab import RESULT_TOKENS, UciTokenizer

if TYPE_CHECKING:
    from tokenizers import Tokenizer

TOKENS_FILE = "tokens.npy"
STARTS_FILE = "starts.npy"
META_FILE = "meta.json"
CHUNK_GAMES = 20_000


class GameEncoder(Protocol):
    """What ``pack_month`` needs from a tokenizer."""

    scheme: str
    pad_id: int
    bos_id: int
    eos_id: int

    def vocab_size(self) -> int: ...

    def vocab_hash(self) -> str: ...

    def encode_game(
        self, uci: str, white_elo: int, black_elo: int, result: str, max_len: int
    ) -> list[int]: ...


class UciGameEncoder:
    """The fixed UCI vocabulary as a ``GameEncoder``."""

    scheme = "uci"

    def __init__(self, tokenizer: UciTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or UciTokenizer()
        self.pad_id = self.tokenizer.pad_id
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id

    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def vocab_hash(self) -> str:
        return _sha256_text(json.dumps(self.tokenizer.ids))

    def encode_game(
        self, uci: str, white_elo: int, black_elo: int, result: str, max_len: int
    ) -> list[int]:
        return self.tokenizer.encode_game(uci, white_elo, black_elo, result, max_len=max_len)


class SanGameEncoder:
    """Char-level SAN: ``[<bos>, chars of '1.e4 e5 ... 1-0', <eos>]`` (no Elo tokens)."""

    scheme = "san"

    def __init__(self, tokenizer: SanCharTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or SanCharTokenizer()
        self.pad_id = self.tokenizer.pad_id
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id

    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def vocab_hash(self) -> str:
        return _sha256_text(json.dumps(self.tokenizer.ids))

    def encode_game(
        self, uci: str, white_elo: int, black_elo: int, result: str, max_len: int
    ) -> list[int]:
        return self.tokenizer.encode(san_text(uci_to_san(uci), result))[:max_len]


class BpeGameEncoder:
    """BPE over spaceless UCI: ``[<bos>, bpe ids..., <result>, <eos>]`` (no Elo tokens)."""

    scheme = "bpe"

    def __init__(self, bpe: Tokenizer) -> None:
        self.bpe = bpe
        self.pad_id = UciTokenizer.pad_id
        self.bos_id = UciTokenizer.bos_id
        self.eos_id = UciTokenizer.eos_id
        for token in UciTokenizer.specials:
            if bpe.token_to_id(token) != UciTokenizer().vocab[token]:
                raise ValueError(f"BPE special token {token} does not share the UCI id")

    def vocab_size(self) -> int:
        return self.bpe.get_vocab_size()

    def vocab_hash(self) -> str:
        return _sha256_text(self.bpe.to_str())

    def encode_game(
        self, uci: str, white_elo: int, black_elo: int, result: str, max_len: int
    ) -> list[int]:
        from rukh.tokenize.bpe import bpe_text

        if result not in RESULT_TOKENS:
            raise ValueError(f"unknown result {result!r}")
        ids = [self.bos_id, *self.bpe.encode(bpe_text(uci)).ids]
        ids.append(self.bpe.token_to_id(RESULT_TOKENS[result]))
        ids.append(self.eos_id)
        return ids[:max_len]


class PackInfo(BaseModel):
    """Contents of ``meta.json``."""

    model_config = ConfigDict(extra="forbid")

    n_games: int
    n_tokens: int
    scheme: str
    vocab_size: int
    vocab_hash: str
    source: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_encoder(scheme: str, bpe: Tokenizer | None = None) -> GameEncoder:
    if scheme == "uci":
        return UciGameEncoder()
    if scheme == "san":
        return SanGameEncoder()
    if scheme == "bpe":
        if bpe is None:
            raise ValueError("scheme 'bpe' needs a trained tokenizer")
        return BpeGameEncoder(bpe)
    raise ValueError(f"unknown scheme {scheme!r}")


def pack_month(games: Path, encoder: GameEncoder, out: Path) -> PackInfo:
    """Encode every game of ``games`` (a UCI parquet) into ``out/tokens.npy`` + ``starts.npy``.

    Games are streamed in chunks so memory stays flat; the token stream is first appended to a
    raw file and then copied into a proper ``.npy`` once its length is known.
    """
    import polars as pl

    if encoder.vocab_size() > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocabulary does not fit in uint16")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "tokens.bin"
    starts: list[int] = []
    n_tokens = 0
    frame = pl.scan_parquet(Path(games).as_posix()).select(
        "uci", "white_elo", "black_elo", "result"
    )
    with raw_path.open("wb") as raw:
        for chunk in frame.collect().iter_slices(CHUNK_GAMES):
            buffer: list[int] = []
            for uci, w_elo, b_elo, result in chunk.iter_rows():
                ids = encoder.encode_game(uci, int(w_elo), int(b_elo), result, max_len=1 << 30)
                starts.append(n_tokens)
                n_tokens += len(ids)
                buffer.extend(ids)
            raw.write(np.asarray(buffer, dtype=np.uint16).tobytes())
    tokens = np.lib.format.open_memmap(
        out / TOKENS_FILE, mode="w+", dtype=np.uint16, shape=(n_tokens,)
    )
    if n_tokens:
        tokens[:] = np.fromfile(raw_path, dtype=np.uint16)
    tokens.flush()
    del tokens
    raw_path.unlink()
    np.save(out / STARTS_FILE, np.asarray(starts, dtype=np.int64))
    info = PackInfo(
        n_games=len(starts),
        n_tokens=n_tokens,
        scheme=encoder.scheme,
        vocab_size=encoder.vocab_size(),
        vocab_hash=encoder.vocab_hash(),
        source=Path(games).name,
    )
    (out / META_FILE).write_text(info.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return info


def read_pack_info(directory: Path) -> PackInfo:
    return PackInfo.model_validate_json((Path(directory) / META_FILE).read_text("utf-8"))


def month_parquet(uci_dir: Path, month: str) -> Path:
    year, mm = month.split("-")
    return Path(uci_dir) / f"year={year}" / f"month={mm}" / "games.parquet"


def pack_scheme(cfg: TokenizeConfig, scheme: str, bpe: Tokenizer | None = None) -> list[Path]:
    """Pack ``train_month`` and ``val_month`` under ``out_dir/<scheme>/{train,val}``."""
    encoder = make_encoder(scheme, bpe)
    uci_dir = resolve(cfg.uci_dir)
    written: list[Path] = []
    for split, month in (("train", cfg.train_month), ("val", cfg.val_month)):
        out = resolve(cfg.out_dir) / scheme / split
        pack_month(month_parquet(uci_dir, month), encoder, out)
        written.append(out / META_FILE)
    return written
