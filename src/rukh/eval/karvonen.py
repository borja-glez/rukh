"""Adam Karvonen's chess nanoGPT as a ``Player``: the public baseline of the results table.

A table of our own models measured by our own harness proves nothing about the harness. One row
has to be a model somebody else built, published and described, put through *exactly* the same
ladder, positions and puzzles as ours -- if that row lands where its author's own numbers say it
should, the harness is credible; if it does not, the difference is the first thing to explain.

The baseline is the 8-layer, 25 M parameter character-level GPT of ``adamkarvonen/chess_llms``
(``lichess_8layers_ckpt_no_optimizer.pt``), trained with Karpathy's nanoGPT on 16 M Lichess games
written as ``;1.e4 e5 2.Nf3 Nc6`` -- one character per token, a ``;`` before every game, no space
after the move number. That format is reproduced here to the character, because the author's
own README warns that performance drops without the ``;``; the model has no Elo header and no
result, so it is asked the same question as ours with less information, which the reader should
keep in mind when comparing rows.

The network is nanoGPT's ``GPT`` reimplemented in a hundred lines rather than imported: the
checkpoint is a ``torch.compile`` state dict with ``_orig_mod.`` prefixes and nothing else, and
a baseline that pulls in a training repository to run inference is a baseline nobody rebuilds.
The vocabulary (``meta.pkl`` in his ``chess_llm_interpretability`` repository) is 32 characters,
pinned here by SHA-256 and by value.
"""

from __future__ import annotations

import hashlib
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rukh import paths
from rukh.eval.qwen_source import DECODER_PLIES, QwenStats, parse_san_answer

log = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_FILE",
    "HF_REPO",
    "META_SHA256",
    "VOCAB",
    "GPT",
    "GPTConfig",
    "KarvonenPlayer",
    "KarvonenSource",
    "ensure_karvonen",
    "karvonen_source",
    "load_karvonen",
    "prefix_board",
    "read_meta",
    "transcript",
]

HF_REPO = "adamkarvonen/chess_llms"
CHECKPOINT_FILE = "lichess_8layers_ckpt_no_optimizer.pt"
LOCAL_DIR = "checkpoints/karvonen-8l"
META_URL = (
    "https://raw.githubusercontent.com/adamkarvonen/chess_llm_interpretability/main/models/meta.pkl"
)
META_SHA256 = "f1121191e401988851de5744fe27fe463c3a086fc8c9a5538ef7fc12162bfb09"
"""The vocabulary file is not on the Hub, so it is fetched from GitHub and pinned by hash."""
VOCAB = " #+-.0123456789;=BKNOQRabcdefghx"
"""``meta.pkl`` by value: character ``i`` of this string is token ``i``. Every SAN move
python-chess writes (``exd5``, ``Nbd2``, ``O-O-O``, ``e8=Q+``, ``Qxf7#``) spells with these."""
STOI = {char: index for index, char in enumerate(VOCAB)}
MAX_NEW_CHARS = 10
"""A SAN move is at most seven characters (``exd8=Q+``); the author's harness also stops at 10."""
COMPILED_PREFIX = "_orig_mod."


@dataclass(frozen=True)
class GPTConfig:
    """nanoGPT's ``model_args`` as the checkpoint carries them; ``dropout`` is unused here."""

    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 1023
    bias: bool = False
    vocab_size: int = 32
    dropout: float = 0.0


class _Attention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        shape = (batch, length, self.n_head, width // self.n_head)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(batch, length, width))


class _MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class _Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = _Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = _MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """Karpathy's nanoGPT, exactly as far as the state dict needs: same names, same shapes."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
                "wpe": nn.Embedding(cfg.block_size, cfg.n_embd),
                "h": nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layer)]),
                "ln_f": nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
            }
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # tied, as nanoGPT ties them

    def forward(self, idx: Tensor) -> Tensor:
        """Logits over the vocabulary at every position of ``idx`` (``batch, length``)."""
        length = idx.shape[1]
        if length > self.cfg.block_size:
            raise ValueError(f"sequence of {length} exceeds block size {self.cfg.block_size}")
        pos = torch.arange(length, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


def strip_compiled(state: dict[str, Any]) -> dict[str, Any]:
    """The state dict without the ``_orig_mod.`` that ``torch.compile`` leaves on every key."""
    return {key.removeprefix(COMPILED_PREFIX): value for key, value in state.items()}


def load_karvonen(checkpoint: Path | str, device: str | None = None) -> GPT:
    """The model of a nanoGPT checkpoint (``model_args`` plus ``model``), in eval mode."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    args = {k: v for k, v in payload["model_args"].items() if k in GPTConfig.__dataclass_fields__}
    model = GPT(GPTConfig(**args))
    model.load_state_dict(strip_compiled(payload["model"]))
    return model.to(device or "cpu").eval()


def read_meta(path: Path | str) -> dict[str, int]:
    """``stoi`` of ``meta.pkl``, refused unless it is byte for byte the pinned file."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != META_SHA256:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned {META_SHA256}")
    stoi = dict(pickle.loads(raw)["stoi"])  # noqa: S301 - the hash above pins the bytes
    if stoi != STOI:
        raise ValueError(f"{path}: vocabulary differs from the one this player writes")
    return stoi


def ensure_karvonen(download: bool = True, local_dir: Path | str | None = None) -> Path:
    """The checkpoint on disk (downloading it and ``meta.pkl`` when allowed), and its path.

    The Hub repository is somebody else's, so this is not a ``rukh pull`` artefact -- nothing here
    is published by the course, nothing can be. ``huggingface_hub`` caches the 100 MB file once.
    """
    directory = Path(local_dir) if local_dir is not None else paths.resolve(LOCAL_DIR)
    checkpoint = directory / CHECKPOINT_FILE
    meta = directory / "meta.pkl"
    if not checkpoint.is_file():
        if not download:
            raise FileNotFoundError(f"{checkpoint} is not on disk (rukh eval karvonen pulls it)")
        import huggingface_hub

        log.info("downloading %s/%s", HF_REPO, CHECKPOINT_FILE)
        directory.mkdir(parents=True, exist_ok=True)
        huggingface_hub.hf_hub_download(HF_REPO, CHECKPOINT_FILE, local_dir=str(directory))
    if not meta.is_file() and download:
        import urllib.request

        log.info("downloading %s", META_URL)
        with urllib.request.urlopen(META_URL, timeout=60) as response:  # noqa: S310 - https, pinned
            meta.write_bytes(response.read())
    if meta.is_file():
        read_meta(meta)
    return checkpoint


def transcript(board: chess.Board) -> str:
    """The game on ``board`` in the author's format: ``;1.e4 e5 2.Nf3`` and, for White, ``3.``.

    When Black is to move the text ends on White's move with no trailing space: the model writes
    the space itself, as it did in training, and a prompt that carried it would be a spelling it
    never saw. Replayed from the root because SAN depends on the position it is written in.
    """
    replay = board.root()
    parts: list[str] = []
    for move in board.move_stack:
        san = replay.san(move)
        parts.append(f"{replay.fullmove_number}.{san}" if replay.turn == chess.WHITE else san)
        replay.push(move)
    if replay.turn == chess.WHITE:
        parts.append(f"{replay.fullmove_number}.")
    return ";" + " ".join(parts)


class KarvonenPlayer:
    """A ``rukh.infer.Player`` that writes SAN one character at a time from the transcript.

    Same accounting as ``QwenPlayer`` (``QwenStats``, ``parse_san_answer``, first legal move as the
    fallback and never a second sample), so its failure breakdown is comparable with that row.
    """

    def __init__(
        self,
        model: GPT,
        stats: QwenStats | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        seed: int | None = None,
        max_new_chars: int = MAX_NEW_CHARS,
        max_plies: int = DECODER_PLIES,
    ) -> None:
        self.model = model
        self.stats = stats if stats is not None else QwenStats()
        self.temperature, self.top_k, self.max_new_chars = temperature, top_k, max_new_chars
        self.max_plies = max_plies
        self.device = next(model.parameters()).device
        self.generator: torch.Generator | None = None
        if seed is not None:
            self.generator = torch.Generator(device="cpu").manual_seed(seed)

    @property
    def limit(self) -> int:
        """Plies this player can carry: the decoder's number, so every row plays the same game."""
        return self.max_plies

    def start(self, white_elo: int, black_elo: int) -> None:
        """Nothing to set: this model has no rating header, which is part of what it is."""

    def observe(self, move: chess.Move) -> None:
        """Nothing to record: the board carries the history and the transcript is built from it."""

    def _next(self, logits: Tensor) -> int:
        if self.temperature <= 0 or self.top_k == 1:
            return int(logits.argmax())
        logits = logits / self.temperature
        if self.top_k is not None and self.top_k < logits.numel():
            cutoff = torch.topk(logits, self.top_k).values[-1]
            logits = logits.masked_fill(logits < cutoff, -math.inf)
        probs = F.softmax(logits, dim=-1).cpu()
        return int(torch.multinomial(probs, 1, generator=self.generator))

    def write(self, prompt: str) -> str:
        """Characters the model appends to ``prompt``, up to the space that ends the move."""
        ids = [STOI[char] for char in prompt]
        written: list[str] = []
        with torch.no_grad():
            for _ in range(self.max_new_chars):
                window = ids[-self.model.cfg.block_size :]
                logits = self.model(torch.tensor([window], device=self.device))[0, -1]
                char = VOCAB[self._next(logits.float())]
                if char == ";" or (char == " " and "".join(written).strip()):
                    break  # a new game, or the end of the move
                written.append(char)
                ids.append(STOI[char])
        return "".join(written)

    def answer(self, board: chess.Board) -> str:
        """The raw continuation the model writes for this position."""
        return self.write(transcript(board))

    def propose(self, board: chess.Board) -> tuple[chess.Move | None, str]:
        """What the model wrote, parsed, with the outcome named. No mask, no rescue."""
        move, outcome = parse_san_answer(board, self.answer(board))
        self.stats.record(outcome)
        return move, outcome

    def choose(self, board: chess.Board) -> tuple[chess.Move | None, bool]:
        """The move to play and whether its own answer was usable; ``QwenPlayer``'s rule exactly."""
        move, _ = self.propose(board)
        if move is not None:
            return move, True
        legal = list(board.legal_moves)
        return (legal[0] if legal else None), False


def prefix_board(board: chess.Board, history: list[int], tok: Any) -> chess.Board:
    """``board`` with the real game in front of it, rebuilt from the decoder's prompt ids.

    A puzzle board starts from a FEN and its history lives only in the tokens the decoder reads.
    This model cannot read a FEN at all -- a transcript that begins at move 23 with no moves is
    text it never saw -- so the game is replayed from the UCI tokens and used when it reaches the
    puzzle's position. When it does not (a line-only parquet), the board is returned as it is and
    the caller counts that the model played blind.
    """
    replay = chess.Board()
    for token in tok.decode(history):
        if token.startswith("<"):
            continue
        try:
            replay.push_uci(token)
        except ValueError:
            return board
    same = replay.fen().split(" ")[:4] == board.fen().split(" ")[:4]
    return replay if same else board


class KarvonenSource:
    """The ``MoveSource`` the puzzle suite expects; ``blind`` counts the prompts with no game."""

    def __init__(self, player: KarvonenPlayer, tok: Any) -> None:
        self.player, self.tok = player, tok
        self.blind = 0

    def __call__(self, board: chess.Board, history: list[int]) -> chess.Move | None:
        prompted = prefix_board(board, history, self.tok)
        if prompted is board:
            self.blind += 1
        move, _ = self.player.choose(prompted)
        return move


def karvonen_source(player: KarvonenPlayer, tok: Any) -> KarvonenSource:
    """Puzzles are scored on the move the model *chooses*, fallback included, as for Qwen."""
    return KarvonenSource(player, tok)
