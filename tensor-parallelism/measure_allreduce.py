"""Measure all-reduce latency and compare against compute time.

Quantifies the communication overhead that limits TP scaling efficiency.

Run:
    torchrun --nproc_per_node=N measure_allreduce.py
"""

import time

import torch
import torch.distributed as dist

from model import D_MODEL, D_FF, N_LAYERS, BATCH_SIZE, SEQ_LEN


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)

    # ----- All-reduce latency across different tensor sizes -----

    sizes = [
        ("Small (1K)", 1024),
        ("Medium (64K)", 65536),
        (f"B={BATCH_SIZE},T={SEQ_LEN},d={D_MODEL}", BATCH_SIZE * SEQ_LEN * D_MODEL),
        ("B=8,T=2048,d=4096", 8 * 2048 * 4096),
    ]

    if rank == 0:
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  ALL-REDUCE LATENCY (TP={ws})")
        print(sep)
        print(f"  {'Size':<28} {'Elements':>12} {'MB':>8} {'Time (us)':>12}")
        print("  " + "-" * 60)

    for name, n in sizes:
        x = torch.randn(n, device=dev, dtype=torch.bfloat16)
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            dist.all_reduce(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        if rank == 0:
            avg_us = sum(times) / len(times) * 1e6
            print(f"  {name:<28} {n:>12,} {n * 2 / 1e6:>8.2f} {avg_us:>12.1f}")

    # ----- Compute vs communication -----

    if rank == 0:
        print(f"\n  COMPUTE vs COMMUNICATION:")

    B, T, D = BATCH_SIZE, SEQ_LEN, D_MODEL
    xm = torch.randn(B * T, D, device=dev, dtype=torch.bfloat16)
    wm = torch.randn(D, D_FF // ws, device=dev, dtype=torch.bfloat16)
    ar = torch.randn(B * T * D, device=dev, dtype=torch.bfloat16)

    for _ in range(5):
        _ = xm @ wm
        dist.all_reduce(ar)
    torch.cuda.synchronize()

    tc = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = xm @ wm
        torch.cuda.synchronize()
        tc.append(time.perf_counter() - t0)

    ta = []
    for _ in range(20):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        dist.all_reduce(ar)
        torch.cuda.synchronize()
        ta.append(time.perf_counter() - t0)

    if rank == 0:
        avg_c = sum(tc) / len(tc) * 1e6
        avg_a = sum(ta) / len(ta) * 1e6
        print(f"  Matmul time:      {avg_c:>10.1f} us")
        print(f"  All-reduce time:  {avg_a:>10.1f} us")
        print(f"  Overhead ratio:   {avg_a / avg_c * 100:>10.1f}%")
        print(f"  TP efficiency:    {avg_c / (avg_c + avg_a) * 100:>10.1f}%")
        print(
            f"  Total comm/step:  ~{12 * avg_a / 1000:.2f} ms "
            f"({N_LAYERS} layers x 2 ARs)"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
