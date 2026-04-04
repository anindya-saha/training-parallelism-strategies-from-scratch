import argparse
import logging
import os
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

DEFAULT_WARMUP = 5
DEFAULT_TRIALS = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 256
DEFAULT_D_MODEL = 512
DEFAULT_D_FF = 2048
DEFAULT_N_LAYERS = 6

ALLREDUCE_SIZES = [
    ("Small (1K)", 1024),
    ("Medium (64K)", 65_536),
    ("B=8, T=256, d=512", 8 * 256 * 512),
    ("B=8, T=2048, d=4096", 8 * 2048 * 4096),
]


def average_us(times: list[float]) -> float:
    return sum(times) / len(times) * 1e6


def benchmark_allreduce(
    rank: int, ws: int, device: torch.device, warmup: int, trials: int
) -> None:
    """Measure all-reduce latency for various tensor sizes."""
    sep = "=" * 70
    dash = "-" * 60

    if rank == 0:
        logger.info("")
        logger.info(sep)
        logger.info("  ALL-REDUCE LATENCY (TP=%d)", ws)
        logger.info(sep)
        logger.info("  %-28s %12s %8s %12s", "Size", "Elements", "MB", "Time (us)")
        logger.info("  %s", dash)

    for name, n in ALLREDUCE_SIZES:
        x = torch.randn(n, device=device, dtype=torch.bfloat16)

        for _ in range(warmup):
            dist.all_reduce(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(trials):
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            dist.all_reduce(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        if rank == 0:
            mb = n * 2 / 1e6  # bfloat16 = 2 bytes
            logger.info(
                "  %-28s %12s %8.2f %12.1f", name, f"{n:,}", mb, average_us(times)
            )

    if rank == 0:
        logger.info(sep)


def benchmark_compute_vs_comm(
    rank: int,
    ws: int,
    device: torch.device,
    warmup: int,
    trials: int,
    B: int,
    T: int,
    D: int,
    D_FF: int,
    n_layers: int,
) -> None:
    """Compare matmul compute time against all-reduce communication time."""
    x = torch.randn(B * T, D, device=device, dtype=torch.bfloat16)
    w = torch.randn(D, D_FF // ws, device=device, dtype=torch.bfloat16)
    ar_buf = torch.randn(B * T * D, device=device, dtype=torch.bfloat16)

    for _ in range(warmup):
        _ = x @ w
        dist.all_reduce(ar_buf)
    torch.cuda.synchronize()

    compute_times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = x @ w
        torch.cuda.synchronize()
        compute_times.append(time.perf_counter() - t0)

    comm_times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        dist.all_reduce(ar_buf)
        torch.cuda.synchronize()
        comm_times.append(time.perf_counter() - t0)

    if rank == 0:
        avg_compute = average_us(compute_times)
        avg_comm = average_us(comm_times)
        allreduce_per_step = n_layers * 2

        logger.info("")
        logger.info(
            "  COMPUTE vs COMMUNICATION (B=%d, T=%d, D=%d, D_FF=%d):", B, T, D, D_FF
        )
        logger.info("  Matmul time:      %10.1f us", avg_compute)
        logger.info("  All-reduce time:  %10.1f us", avg_comm)
        logger.info("  Overhead ratio:   %10.1f%%", avg_comm / avg_compute * 100)
        logger.info(
            "  TP efficiency:    %10.1f%%", avg_compute / (avg_compute + avg_comm) * 100
        )
        logger.info(
            "  Total comm/step:  ~%.2f ms (%d layers x 2 ARs = %d/step)",
            allreduce_per_step * avg_comm / 1000,
            n_layers,
            allreduce_per_step,
        )
        logger.info("")


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure all-reduce latency and TP overhead"
    )
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--d-ff", type=int, default=DEFAULT_D_FF)
    p.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
    )

    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(device)

    rank = dist.get_rank()
    ws = dist.get_world_size()

    benchmark_allreduce(rank, ws, device, args.warmup, args.trials)
    benchmark_compute_vs_comm(
        rank,
        ws,
        device,
        args.warmup,
        args.trials,
        B=args.batch_size,
        T=args.seq_len,
        D=args.d_model,
        D_FF=args.d_ff,
        n_layers=args.n_layers,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
