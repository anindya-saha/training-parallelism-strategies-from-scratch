import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_mb(device=0) -> float:
    return torch.cuda.memory_allocated(device) / 1024 / 1024


def get_gpu_peak_memory_mb(device=0) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024
