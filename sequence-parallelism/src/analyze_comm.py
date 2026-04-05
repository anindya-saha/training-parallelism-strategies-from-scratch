"""Analyze communication patterns: Vanilla TP vs TP+SP.

Key insight: ALL-REDUCE = REDUCE-SCATTER + ALL-GATHER
So TP+SP has the same communication volume as Vanilla TP!
"""

import torch
import torch.distributed as dist
import time
import sys
import os

from config import *


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank = dist.get_rank()
    ws = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    B, S, h = BATCH_SIZE, SEQ_LEN, D_MODEL
    S_local = S // ws

    if rank == 0:
        print()
        print("=" * 80)
        print(f"  COMMUNICATION ANALYSIS: Vanilla TP vs TP + SP  (N={ws} GPUs)")
        print("=" * 80)

    # Create test tensors
    full_tensor = torch.randn(B, S, h, device=device, dtype=torch.bfloat16)
    local_tensor = torch.randn(B, S_local, h, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        x = full_tensor.clone()
        dist.all_reduce(x)
        chunks = list(full_tensor.split(S_local, dim=1))
        out = torch.empty_like(local_tensor)
        dist.reduce_scatter(out, chunks)
        gathered = [torch.empty_like(local_tensor) for _ in range(ws)]
        dist.all_gather(gathered, local_tensor)

    torch.cuda.synchronize()
    dist.barrier()

    # Time ALL-REDUCE
    ar_times = []
    for _ in range(20):
        x = full_tensor.clone()
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(x)
        torch.cuda.synchronize()
        ar_times.append(time.perf_counter() - t0)

    # Time REDUCE-SCATTER
    rs_times = []
    for _ in range(20):
        x = full_tensor.clone()
        chunks = list(x.split(S_local, dim=1))
        out = torch.empty(B, S_local, h, device=device, dtype=torch.bfloat16)
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.reduce_scatter(out, chunks)
        torch.cuda.synchronize()
        rs_times.append(time.perf_counter() - t0)

    # Time ALL-GATHER
    ag_times = []
    for _ in range(20):
        gathered = [torch.empty_like(local_tensor) for _ in range(ws)]
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_gather(gathered, local_tensor)
        torch.cuda.synchronize()
        ag_times.append(time.perf_counter() - t0)

    # Skip first few (warmup)
    avg_ar = sum(ar_times[5:]) / len(ar_times[5:]) * 1000
    avg_rs = sum(rs_times[5:]) / len(rs_times[5:]) * 1000
    avg_ag = sum(ag_times[5:]) / len(ag_times[5:]) * 1000

    if rank == 0:
        data_elements = B * S * h
        data_mb = data_elements * 2 / 1024 / 1024  # BF16

        print()
        print(
            f"  Tensor shape: (B={B}, S={S}, h={h}) = {data_elements:,} elements = {data_mb:.2f} MB"
        )
        print()

        print("  " + "-" * 70)
        print(f"  {'Operation':<30} {'Time (ms)':<15} {'Notes'}")
        print("  " + "-" * 70)
        print(f"  {'ALL-REDUCE':<30} {avg_ar:<15.3f} Used by Vanilla TP")
        print(f"  {'REDUCE-SCATTER':<30} {avg_rs:<15.3f} TP to SP transition")
        print(f"  {'ALL-GATHER':<30} {avg_ag:<15.3f} SP to TP transition")
        print(f"  {'RS + AG (combined)':<30} {avg_rs + avg_ag:<15.3f} Used by TP+SP")
        print("  " + "-" * 70)

        print()
        print("  RESULT:")
        print(f"    ALL-REDUCE time:              {avg_ar:.3f} ms")
        print(f"    REDUCE-SCATTER + ALL-GATHER:  {avg_rs + avg_ag:.3f} ms")
        ratio = (avg_rs + avg_ag) / avg_ar
        print(f"    Ratio: {ratio:.2f}x", end="")
        if 0.8 <= ratio <= 1.2:
            print("  (approximately equal, as expected!)")
        else:
            print()

        print()
        print("=" * 80)
        print("  PER-LAYER COMMUNICATION BREAKDOWN")
        print("=" * 80)
        print()
        print("  Vanilla TP (per transformer block):")
        print(f"    After W_o:  ALL-REDUCE  (B,S,h) = {data_mb:.2f} MB")
        print(f"    After W2:   ALL-REDUCE  (B,S,h) = {data_mb:.2f} MB")
        print(f"    TOTAL: 2 x ALL-REDUCE = 2 x {data_mb:.2f} MB = {2*data_mb:.2f} MB")
        print()
        print("  TP + SP (per transformer block):")
        print(f"    Before Q,K,V: ALL-GATHER     (B,S/N,h) -> (B,S,h)")
        print(f"    After W_o:    REDUCE-SCATTER (B,S,h) -> (B,S/N,h)")
        print(f"    Before W1:    ALL-GATHER     (B,S/N,h) -> (B,S,h)")
        print(f"    After W2:     REDUCE-SCATTER (B,S,h) -> (B,S/N,h)")
        print(f"    TOTAL: 2 x (RS + AG) = 2 x {data_mb:.2f} MB = {2*data_mb:.2f} MB")
        print()
        print("  CONCLUSION: Same communication volume!")
        print("    ALL-REDUCE internally = REDUCE-SCATTER + ALL-GATHER")
        print(
            "    TP+SP just separates these to do useful work (LayerNorm/Dropout) in between"
        )
        print()

        print("=" * 80)
        print("  WHY THIS MATTERS")
        print("=" * 80)
        print("""
  Vanilla TP:
    [matmul] -> [ALL-REDUCE] -> [LayerNorm on (B,S,h)] -> ...
                 (fused)        full tensor, wasted memory!

  TP + SP:
    [matmul] -> [REDUCE-SCATTER] -> [LayerNorm on (B,S/N,h)] -> [ALL-GATHER] -> ...
                 (explicit)         smaller tensor, MEMORY SAVED!  (explicit)

  Same bytes sent over the network, but:
  - LayerNorm operates on 1/N of the data
  - Dropout operates on 1/N of the data  
  - Residual connections store 1/N of the data
  
  Result: ~30-50% less activation memory, enabling:
  - Larger batch sizes
  - Longer sequences
  - Or just less memory pressure
        """)
        print("=" * 80)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
