"""Toy MLP for 2-rank pipeline-parallel tutorial scripts (minimal activations).

The model is defined as a **single class** — just like a real HuggingFace or Megatron model.
Pipeline partitioning is done externally by ``get_stage()``, which builds the **full**
``ToyModel`` on ``meta``, reads ``list(full_model.layers)``, and extracts a contiguous
slice for a given rank. No rank ever materializes parameters it does not own. You always
have the whole module definition first (class + state); the ordered list is discovered
from that object, not shipped in from outside.

For a **small GPT**, see ``model_gpt.StandardGPT`` with ``.layers`` plus
``model_gpt.get_stage`` (same ``list(full_model.layers)`` slice idea as below) and
``naive.py`` / ``gpipe.py`` / ``pipedream.py``.

Compute graph (full model):
  x -> embed -> block_0 -> block_1 -> head -> z

With 2 pipeline stages:
  Stage 0 (rank 0): embed, block_0
  Stage 1 (rank 1): block_1, head

This mirrors how production frameworks partition models:
  - DeepSpeed ``PipelineModule``: takes a flat layer list, splits by index
  - Megatron-LM: marks cut points between transformer blocks
  - HuggingFace Accelerate ``device_map``: assigns module children to devices
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ToyModel(nn.Module):
    """Small MLP defined as an ordered layer list — easy to partition."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(dim, dim),  # "embed"
                nn.ReLU(),            # "block_0"
                nn.Linear(dim, dim),  # "block_1"  (also serves as "head")
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def log_model_info(model: nn.Module, label: str = "Model") -> None:
    """Log module structure and parameter shapes."""
    logger.info("%s repr:\n%s", label, model)
    total_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        logger.info(
            "  %s: %r shape=%s dtype=%s (%d params)",
            label, name, tuple(param.shape), param.dtype, n,
        )
    logger.info("%s total parameters: %d", label, total_params)


def get_stage(
    dim: int, rank: int, world_size: int, device: torch.device
) -> nn.Sequential:
    """Partition ``ToyModel`` layers across ranks and return the local stage.

    Each rank only materializes its own slice of the layer list on ``device``.
    This is analogous to how DeepSpeed splits a flat layer list by index:

        all_layers = [embed, block_0, block_1, ..., block_N, head]
        my_layers   = all_layers[start:end]   # contiguous slice for this rank

    The full model is constructed on the ``meta`` device (no memory allocated),
    sliced, then only the local shard is moved to the real device.
    """
    with torch.device("meta"):
        full_model = ToyModel(dim)

    if rank == 0:
        log_model_info(full_model, label="Full model")

    all_layers = list(full_model.layers)
    n_layers = len(all_layers)

    # even split (production systems may use cost-based balancing)
    chunk = (n_layers + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n_layers)
    local_layers = all_layers[start:end]

    logger.info(
        "Rank %d: stage owns layers [%d:%d] of %d", rank, start, end, n_layers
    )

    # materialize only the local layers on the real device
    stage = nn.Sequential(*local_layers)
    stage = stage.to_empty(device=device)
    stage.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    log_model_info(stage, label=f"Rank {rank} Stage")

    return stage
