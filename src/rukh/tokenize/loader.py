"""``PackedDataset`` and ``make_loader``: ``<bos>``-aligned windows over a packed token stream."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rukh.tokenize.pack import STARTS_FILE, TOKENS_FILE, PackInfo, read_pack_info

IGNORE_INDEX = 0  # ``<pad>`` in every scheme; the loss must use ``ignore_index=0``.


class PackedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Windows of ``block`` tokens over ``tokens.npy``.

    With ``start_at_game`` every item starts at a ``<bos>`` (item ``i`` is game ``i``);
    otherwise the stream is cut into consecutive blocks. A window that runs past the end of
    the stream is padded with ``<pad>`` (0). Each item is ``(x, y)`` with ``y = x[1:] + [<pad>]``.
    """

    def __init__(self, directory: Path, block: int = 200, start_at_game: bool = True) -> None:
        if block < 2:
            raise ValueError("block must be at least 2")
        self.directory = Path(directory)
        self.block = block
        self.start_at_game = start_at_game
        self.info: PackInfo = read_pack_info(self.directory)
        self.tokens = np.load(self.directory / TOKENS_FILE, mmap_mode="r")
        self.starts = np.load(self.directory / STARTS_FILE)
        self.pad_id = 0

    def __len__(self) -> int:
        if self.start_at_game:
            return int(len(self.starts))
        return max(1, int(len(self.tokens)) // self.block) if len(self.tokens) else 0

    def window(self, index: int) -> np.ndarray:
        """``block + 1`` tokens from the window origin, padded with ``<pad>`` past the end."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        origin = int(self.starts[index]) if self.start_at_game else index * self.block
        chunk = np.asarray(self.tokens[origin : origin + self.block + 1], dtype=np.int64)
        if len(chunk) < self.block + 1:
            chunk = np.concatenate(
                [chunk, np.full(self.block + 1 - len(chunk), self.pad_id, dtype=np.int64)]
            )
        return chunk

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.window(index)
        x = torch.from_numpy(chunk[: self.block].copy())
        y = torch.from_numpy(chunk[1 : self.block + 1].copy())
        y[-1] = self.pad_id
        return x, y


def make_loader(
    dataset: PackedDataset,
    batch_size: int,
    seed: int = 0,
    workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """A ``DataLoader`` whose shuffling is fully determined by ``seed``."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        generator=generator,
        persistent_workers=workers > 0,
    )
