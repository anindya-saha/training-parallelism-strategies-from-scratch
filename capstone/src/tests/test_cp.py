"""
Test Context Parallelism (Ring Attention) correctness.
Run: torchrun --standalone --nproc_per_node=2 tests/test_cp.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import functional as F


def test_ring_attention():
    """Verify ring attention matches standard causal attention."""
    from parallel.context import ring_attention_forward

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank}'
    torch.manual_seed(42)

    B, T_full, n_head, head_dim = 2, 32, 4, 16
    T_local = T_full // world_size

    # Create full Q, K, V on all ranks
    q_full = torch.randn(B, T_full, n_head, head_dim, device=device)
    k_full = torch.randn(B, T_full, n_head, head_dim, device=device)
    v_full = torch.randn(B, T_full, n_head, head_dim, device=device)
    dist.broadcast(q_full, src=0)
    dist.broadcast(k_full, src=0)
    dist.broadcast(v_full, src=0)

    # Reference: standard causal attention on full sequence
    q_ref = q_full.transpose(1, 2)  # (B, n_head, T_full, head_dim)
    k_ref = k_full.transpose(1, 2)
    v_ref = v_full.transpose(1, 2)
    ref_out = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    # Extract this rank's chunk from reference
    ref_local = ref_out[:, :, rank * T_local:(rank + 1) * T_local, :]

    # Ring attention on local chunks
    q_local = q_full[:, rank * T_local:(rank + 1) * T_local]
    k_local = k_full[:, rank * T_local:(rank + 1) * T_local]
    v_local = v_full[:, rank * T_local:(rank + 1) * T_local]

    cp_group = dist.new_group(list(range(world_size)))
    ring_out = ring_attention_forward(q_local, k_local, v_local, cp_group)
    # ring_out shape: (B, n_head, T_local, head_dim)

    if torch.allclose(ref_local, ring_out.float(), atol=1e-4):
        if rank == 0:
            print("PASS: Ring attention matches standard causal attention")
    else:
        max_diff = (ref_local - ring_out.float()).abs().max().item()
        if rank == 0:
            print(f"FAIL: Ring attention max diff = {max_diff}")


if __name__ == "__main__":
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    test_ring_attention()

    destroy_process_group()
    if int(os.environ.get('RANK', 0)) == 0:
        print("\nCP tests complete.")
