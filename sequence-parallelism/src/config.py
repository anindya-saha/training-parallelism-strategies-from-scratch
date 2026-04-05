"""Shared configuration for Sequence Parallelism tutorial."""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------


D_MODEL = 1024
N_HEADS = 16
D_HEAD = D_MODEL // N_HEADS  # 64
D_FF = 4096
N_LAYERS = 8
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 1024

# ---------------------------------------------------------------------------
# Benchmark Configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
SEQ_LEN = 512
NUM_WARMUP = 3
NUM_BENCHMARK = 10


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_mb(device: int = 0) -> float:
    return torch.cuda.memory_allocated(device) / 1024 / 1024


def get_gpu_peak_memory_mb(device: int = 0) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


def reset_memory_stats(device: int = 0) -> None:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
