"""
Test Expert Parallelism correctness.
Run: torchrun --standalone --nproc_per_node=2 tests/test_ep.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from config import ModelConfig


def test_router():
    """Test TopKRouter produces valid outputs."""
    from parallel.expert import TopKRouter

    rank = dist.get_rank()
    device = f'cuda:{rank}'
    torch.manual_seed(42)

    n_embd, num_experts, top_k = 64, 8, 2
    num_tokens = 32

    router = TopKRouter(n_embd, num_experts, top_k).to(device)
    x = torch.randn(num_tokens, n_embd, device=device)

    indices, weights, aux_loss = router(x)

    # Check shapes
    assert indices.shape == (num_tokens, top_k), f"Expected ({num_tokens}, {top_k}), got {indices.shape}"
    assert weights.shape == (num_tokens, top_k), f"Expected ({num_tokens}, {top_k}), got {weights.shape}"

    # Check weights sum to 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f"Weights don't sum to 1: {weight_sums}"

    # Check indices are valid
    assert (indices >= 0).all() and (indices < num_experts).all(), \
        f"Invalid expert indices: min={indices.min()}, max={indices.max()}"

    # Check aux_loss is a scalar
    assert aux_loss.dim() == 0, f"aux_loss should be scalar, got shape {aux_loss.shape}"

    if rank == 0:
        print(f"PASS: TopKRouter outputs valid (indices range: {indices.min()}-{indices.max()}, "
              f"aux_loss: {aux_loss.item():.4f})")


def test_moe_local():
    """Test MoE layer in local mode (no EP)."""
    from parallel.expert import MoELayer

    rank = dist.get_rank()
    device = f'cuda:{rank}'
    torch.manual_seed(42)

    config = ModelConfig(n_embd=64, num_experts=4, moe_top_k=2)
    B, T = 2, 16

    moe = MoELayer(config, ep_group=None).to(device)
    x = torch.randn(B, T, config.n_embd, device=device)

    output, aux_loss = moe(x)

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    assert aux_loss.dim() == 0, f"aux_loss should be scalar"

    if rank == 0:
        print(f"PASS: MoE local forward (output shape: {output.shape}, aux_loss: {aux_loss.item():.4f})")


def test_moe_ep():
    """Test MoE layer with Expert Parallelism."""
    from parallel.expert import MoELayer

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank}'
    torch.manual_seed(42)

    config = ModelConfig(n_embd=64, num_experts=4, moe_top_k=2)
    B, T = 2, 16

    ep_group = dist.new_group(list(range(world_size)))

    moe_ep = MoELayer(config, ep_group=ep_group).to(device)
    x = torch.randn(B, T, config.n_embd, device=device)
    dist.broadcast(x, src=0)

    output_ep, aux_loss_ep = moe_ep(x)

    assert output_ep.shape == x.shape, f"EP output shape mismatch: {output_ep.shape}"

    if rank == 0:
        print(f"PASS: MoE EP forward (output shape: {output_ep.shape}, aux_loss: {aux_loss_ep.item():.4f})")


if __name__ == "__main__":
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    test_router()
    test_moe_local()
    test_moe_ep()

    destroy_process_group()
    if int(os.environ.get('RANK', 0)) == 0:
        print("\nEP tests complete.")
