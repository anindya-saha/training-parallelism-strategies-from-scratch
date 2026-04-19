"""
Test Tensor Parallelism correctness.
Run: torchrun --standalone --nproc_per_node=2 tests/test_tp.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group


def test_column_parallel():
    """Verify ColumnParallelLinear produces same output as nn.Linear."""
    from parallel.tensor import ColumnParallelLinear

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank}'
    torch.manual_seed(42)

    in_features, out_features = 64, 128
    B, T = 2, 16

    # Create reference linear on rank 0
    ref_linear = nn.Linear(in_features, out_features).to(device)
    x = torch.randn(B, T, in_features, device=device)
    # Broadcast input and weights to all ranks
    dist.broadcast(x, src=0)

    # Create ColumnParallel version
    tp_group = dist.new_group(list(range(world_size)))
    col_linear = ColumnParallelLinear(in_features, out_features, tp_group=tp_group, gather_output=True).to(device)

    # Copy weight slices from reference
    with torch.no_grad():
        chunk_size = out_features // world_size
        col_linear.weight.copy_(ref_linear.weight[rank * chunk_size:(rank + 1) * chunk_size])
        if col_linear.bias is not None:
            col_linear.bias.copy_(ref_linear.bias[rank * chunk_size:(rank + 1) * chunk_size])

    # Compare outputs
    ref_out = ref_linear(x)
    col_out = col_linear(x)

    if torch.allclose(ref_out, col_out, atol=1e-5):
        if rank == 0:
            print("PASS: ColumnParallelLinear matches nn.Linear")
    else:
        max_diff = (ref_out - col_out).abs().max().item()
        if rank == 0:
            print(f"FAIL: ColumnParallelLinear max diff = {max_diff}")


def test_row_parallel():
    """Verify RowParallelLinear produces same output as nn.Linear."""
    from parallel.tensor import RowParallelLinear

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank}'
    torch.manual_seed(42)

    in_features, out_features = 128, 64
    B, T = 2, 16

    # Create reference
    ref_linear = nn.Linear(in_features, out_features).to(device)
    x = torch.randn(B, T, in_features, device=device)
    dist.broadcast(x, src=0)

    # Create RowParallel version
    tp_group = dist.new_group(list(range(world_size)))
    row_linear = RowParallelLinear(in_features, out_features, tp_group=tp_group).to(device)

    # Copy weight slices
    with torch.no_grad():
        chunk_size = in_features // world_size
        row_linear.weight.copy_(ref_linear.weight[:, rank * chunk_size:(rank + 1) * chunk_size])
        if row_linear.bias is not None:
            row_linear.bias.copy_(ref_linear.bias)

    # Split input along last dim (simulating output of ColumnParallel)
    x_local = x[:, :, rank * chunk_size:(rank + 1) * chunk_size]

    ref_out = ref_linear(x)
    row_out = row_linear(x_local)

    if torch.allclose(ref_out, row_out, atol=1e-5):
        if rank == 0:
            print("PASS: RowParallelLinear matches nn.Linear")
    else:
        max_diff = (ref_out - row_out).abs().max().item()
        if rank == 0:
            print(f"FAIL: RowParallelLinear max diff = {max_diff}")


if __name__ == "__main__":
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    test_column_parallel()
    test_row_parallel()

    destroy_process_group()
    if int(os.environ.get('RANK', 0)) == 0:
        print("\nTP tests complete.")
