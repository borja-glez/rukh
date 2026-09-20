"""Pack encoded games into one ``uint16`` stream (``tokens.npy``) plus a game index.

The stream is the concatenation of every encoded game (``<bos> ... <eos>``), untruncated;
``starts.npy`` holds the offset of each ``<bos>`` so a loader can cut ``block``-sized windows
aligned to game starts (D-019). ``meta.json`` records counts, the scheme and a hash of the
vocabulary so a stale pack is never mixed with a different tokenizer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pyarrow.parquet as pq
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
COPY_TOKENS = (64 << 20) // 2  # 64 MiB of uint16 per copy into the memmap


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


@dataclass(frozen=True)
class GameSlice:
    """A contiguous run of games inside one UCI parquet.

    ``skip`` games are dropped from the front and at most ``take`` are kept (``None`` = to the
    end). It exists so one month can feed both splits: the validation slice is the first
    ``val_games`` rows and training gets the remainder, instead of a whole month sitting idle
    as validation when validation only ever reads a few hundred thousand tokens per eval.
    """

    path: Path
    skip: int = 0
    take: int | None = None

    def label(self) -> str:
        """``month=02/games.parquet[100000:]``: what ``meta.json`` records as the source."""
        name = f"{self.path.parent.name}/{self.path.name}"
        if not self.skip and self.take is None:
            return name
        end = "" if self.take is None else str(self.skip + self.take)
        return f"{name}[{self.skip}:{end}]"


def pack_month(games: Path, encoder: GameEncoder, out: Path) -> PackInfo:
    """Encode every game of ``games`` (a UCI parquet) into ``out/tokens.npy`` + ``starts.npy``."""
    return pack_games([GameSlice(Path(games))], encoder, out)


def pack_games(sources: Sequence[GameSlice], encoder: GameEncoder, out: Path) -> PackInfo:
    """Encode every game of ``sources``, in order, into ``out/tokens.npy`` + ``starts.npy``.

    Each parquet is read row group by row group and the token stream is appended to a raw file;
    once its length is known the raw file is copied into the ``.npy`` memmap in 64 MiB chunks,
    so neither the games nor the whole stream are ever held in memory at once. Several sources
    concatenate into a single stream, which is why ``starts.npy`` is written last: its offsets
    are absolute over the concatenation.
    """
    if encoder.vocab_size() > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocabulary does not fit in uint16")
    if not sources:
        raise ValueError("pack_games needs at least one source")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "tokens.bin"
    starts: list[int] = []
    n_tokens = 0
    with raw_path.open("wb") as raw:
        for source in sources:
            reader = pq.ParquetFile(source.path)
            seen = 0
            taken = 0
            for batch in reader.iter_batches(
                batch_size=CHUNK_GAMES, columns=["uci", "white_elo", "black_elo", "result"]
            ):
                rows_here = batch.num_rows
                lo = min(max(source.skip - seen, 0), rows_here)
                hi = rows_here
                if source.take is not None:
                    hi = min(hi, lo + source.take - taken)
                seen += rows_here
                if hi <= lo:
                    if source.take is not None and taken >= source.take:
                        break
                    continue
                batch = batch.slice(lo, hi - lo)
                taken += hi - lo
                buffer: list[int] = []
                rows = zip(
                    batch.column("uci").to_pylist(),
                    batch.column("white_elo").to_pylist(),
                    batch.column("black_elo").to_pylist(),
                    batch.column("result").to_pylist(),
                    strict=True,
                )
                for uci, w_elo, b_elo, result in rows:
                    ids = encoder.encode_game(uci, int(w_elo), int(b_elo), result, max_len=1 << 30)
                    starts.append(n_tokens)
                    n_tokens += len(ids)
                    buffer.extend(ids)
                raw.write(np.asarray(buffer, dtype=np.uint16).tobytes())
                if source.take is not None and taken >= source.take:
                    break
    tokens = np.lib.format.open_memmap(
        out / TOKENS_FILE, mode="w+", dtype=np.uint16, shape=(n_tokens,)
    )
    _copy_raw(raw_path, tokens, n_tokens)
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
        source="+".join(source.label() for source in sources),
    )
    (out / META_FILE).write_text(info.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return info


def _copy_raw(raw_path: Path, tokens: np.ndarray, n_tokens: int) -> None:
    """Copy the raw token file into the memmap in ``COPY_TOKENS``-sized chunks."""
    offset = 0
    with raw_path.open("rb") as raw:
        while offset < n_tokens:
            chunk = np.fromfile(raw, dtype=np.uint16, count=min(COPY_TOKENS, n_tokens - offset))
            if not len(chunk):
                raise OSError(f"{raw_path}: expected {n_tokens} tokens, got {offset}")
            tokens[offset : offset + len(chunk)] = chunk
            offset += len(chunk)


def read_pack_info(directory: Path) -> PackInfo:
    return PackInfo.model_validate_json((Path(directory) / META_FILE).read_text("utf-8"))


def month_parquet(uci_dir: Path, month: str) -> Path:
    year, mm = month.split("-")
    return Path(uci_dir) / f"year={year}" / f"month={mm}" / "games.parquet"


def split_sources(cfg: TokenizeConfig) -> dict[str, list[GameSlice]]:
    """The parquet slices behind each split, as ``pack_scheme`` will read them.

    ``train_months`` are taken whole. ``val_month`` is cut in two: its first ``val_games`` games
    are the validation set and everything after them joins training. With ``val_games = 0`` the
    whole month is validation, which is what P1 and P2 did -- and what left 238 M tokens unused,
    since an eval reads 50 batches and not a month (D-062).

    ``extra_train_parquets`` appends UCI parquets that live outside the ``year=/month=`` layout,
    which is how the Elite Database (2200+ only) joins the corpus without pretending to be a
    Lichess month.
    """
    uci_dir = resolve(cfg.uci_dir)
    train = [GameSlice(month_parquet(uci_dir, month)) for month in cfg.train_months]
    val_parquet = month_parquet(uci_dir, cfg.val_month)
    if cfg.val_games:
        val = [GameSlice(val_parquet, take=cfg.val_games)]
        train.append(GameSlice(val_parquet, skip=cfg.val_games))
    else:
        val = [GameSlice(val_parquet)]
    train += [GameSlice(resolve(extra)) for extra in cfg.extra_train_parquets]
    return {"train": train, "val": val}


def pack_scheme(cfg: TokenizeConfig, scheme: str, bpe: Tokenizer | None = None) -> list[Path]:
    """Pack the ``train`` and ``val`` splits under ``out_dir/<scheme>/{train,val}``."""
    encoder = make_encoder(scheme, bpe)
    written: list[Path] = []
    for split, sources in split_sources(cfg).items():
        out = resolve(cfg.out_dir) / scheme / split
        pack_games(sources, encoder, out)
        written.append(out / META_FILE)
    return written
