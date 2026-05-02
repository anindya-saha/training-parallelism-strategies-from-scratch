"""TinyStories data pipeline backed by memory-mapped binary files.

The dataset files (train.bin, validation.bin) contain GPT-2 tokenized text
stored as uint16 numpy arrays. We use np.memmap to avoid loading the full
901MB training file into RAM -- the OS pages data in on demand.

Usage:
    from data import create_dataloader

    train_loader = create_dataloader(
        data_path="train.bin", seq_len=256, batch_size=8,
        dp_rank=0, dp_size=1,
    )
    for input_ids, labels in train_loader:
        ...
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class MemmapDataset(Dataset):
    """Token dataset backed by a memory-mapped binary file.

    Each sample is a (input_ids, labels) pair where labels are input_ids
    shifted right by one position (standard autoregressive LM objective).
    """

    def __init__(self, data_path: str, seq_len: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])  # (seq_len,)
        y = torch.from_numpy(chunk[1:])   # (seq_len,)
        return x, y


def create_dataloader(
    data_path: str,
    seq_len: int,
    batch_size: int,
    dp_rank: int = 0,
    dp_size: int = 1,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 2,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a DataLoader with optional distributed sampling for DP.

    Returns (dataloader, sampler) -- caller must call sampler.set_epoch()
    each epoch when dp_size > 1.
    """
    dataset = MemmapDataset(data_path, seq_len)

    sampler = None
    if dp_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=shuffle,
            seed=seed,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler
