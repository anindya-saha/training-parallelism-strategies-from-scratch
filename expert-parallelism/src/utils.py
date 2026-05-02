import torch
import torch.nn as nn

# ================================================================
# Utilities
# ================================================================


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_active_parameters(model: nn.Module, num_experts: int, top_k: int) -> int:
    """Count parameters activated per forward pass in a MoE model.

    Non-expert params are always active. Expert params are scaled by
    top_k / num_experts since only top_k of num_experts are used per token.
    """
    expert_params = 0
    non_expert_params = 0
    for name, p in model.named_parameters():
        if ".experts." in name:
            expert_params += p.numel()
        else:
            non_expert_params += p.numel()
    active_expert_params = int(expert_params * top_k / num_experts)
    return non_expert_params + active_expert_params


def get_gpu_memory_mb(device=0) -> float:
    return torch.cuda.memory_allocated(device) / 1024 / 1024


def get_gpu_peak_memory_mb(device=0) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024
