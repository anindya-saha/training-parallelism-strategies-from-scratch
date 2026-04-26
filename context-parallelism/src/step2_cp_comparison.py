"""Step 2: Context Parallelism Comparison

Compares:
  1. Standard Attention: Full (S x S) attention matrix
  2. CP Attention: Smaller (S/CP x S) attention matrix per GPU

Memory savings come from each GPU only computing attention
for its local queries (S/CP) instead of all queries (S).
"""

import json
import logging
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    ws = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Config
    B, H, D = 4, 16, 64  # batch, heads, head_dim

    if rank == 0:
        logger.info("")
        logger.info("=" * 65)
        logger.info("  CONTEXT PARALLELISM COMPARISON")
        logger.info("=" * 65)
        logger.info("  GPUs: %s", ws)
        logger.info("  Batch: %s, Heads: %s, Head dim: %s", B, H, D)
        logger.info("=" * 65)
        logger.info("")

    all_results = {}

    for S in [1024, 2048, 4096]:
        S_local = S // ws

        if rank == 0:
            logger.info("--- Sequence Length: %s ---", S)

        # ============================================================
        # TEST 1: Standard Attention (No CP)
        # Each GPU computes full (S x S) attention
        # ============================================================
        Q = torch.randn(B, H, S, D, device=device)
        K = torch.randn(B, H, S, D, device=device)
        V = torch.randn(B, H, S, D, device=device)

        # Warmup
        for _ in range(3):
            attn = (Q @ K.transpose(-2, -1)) / math.sqrt(D)
            out = F.softmax(attn, dim=-1) @ V
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(10):
            attn = (Q @ K.transpose(-2, -1)) / math.sqrt(D)
            out = F.softmax(attn, dim=-1) @ V
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        mem_no_cp = torch.cuda.max_memory_allocated(device) / 1024**2
        time_no_cp = (t1 - t0) / 10 * 1000

        del Q, K, V, attn, out
        torch.cuda.empty_cache()

        # ============================================================
        # TEST 2: CP-Style Attention
        # Each GPU computes (S/CP x S) attention for its local queries
        # ============================================================
        Q_local = torch.randn(B, H, S_local, D, device=device)
        K_full = torch.randn(B, H, S, D, device=device)
        V_full = torch.randn(B, H, S, D, device=device)

        # Warmup
        for _ in range(3):
            attn = (Q_local @ K_full.transpose(-2, -1)) / math.sqrt(D)
            out = F.softmax(attn, dim=-1) @ V_full
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(10):
            attn = (Q_local @ K_full.transpose(-2, -1)) / math.sqrt(D)
            out = F.softmax(attn, dim=-1) @ V_full
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        mem_cp = torch.cuda.max_memory_allocated(device) / 1024**2
        time_cp = (t1 - t0) / 10 * 1000

        del Q_local, K_full, V_full, attn, out
        torch.cuda.empty_cache()

        if rank == 0:
            reduction = (mem_no_cp - mem_cp) / mem_no_cp * 100
            logger.info(
                "  No CP:   attn=(%sx%s)       mem=%7.1f MB  time=%6.2f ms",
                S,
                S,
                mem_no_cp,
                time_no_cp,
            )
            logger.info(
                "  CP=%s:    attn=(%sx%s)     mem=%7.1f MB  time=%6.2f ms",
                ws,
                S_local,
                S,
                mem_cp,
                time_cp,
            )
            logger.info("  Reduction: %.1f%%", reduction)
            logger.info("")

            all_results[str(S)] = {
                "no_cp": {"mem_mb": round(mem_no_cp, 1), "time_ms": round(time_no_cp, 2)},
                "with_cp": {"mem_mb": round(mem_cp, 1), "time_ms": round(time_cp, 2)},
                "reduction_pct": round(reduction, 1),
            }

    if rank == 0:
        with open("outputs/cp_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        logger.info("=" * 65)
        logger.info("  SUMMARY")
        logger.info("=" * 65)
        logger.info("  %-8s %12s %12s %12s", "Seq", "No CP Mem", "CP Mem", "Reduction")
        logger.info("  %s", "-" * 44)
        for s, r in all_results.items():
            logger.info(
                "  %-8s %9.1f MB %9.1f MB %9.1f%%",
                s,
                r["no_cp"]["mem_mb"],
                r["with_cp"]["mem_mb"],
                r["reduction_pct"],
            )

        logger.info("")
        logger.info("=" * 65)
        logger.info("  KEY INSIGHT")
        logger.info("=" * 65)
        logger.info(
            """
  Standard Attention:
    Each GPU: Q @ K.T  ->  (%s, %s, S, S) attention matrix
    Memory: O(S^2)

  Context Parallel Attention (CP=%s):
    Each GPU: Q_local @ K_full.T  ->  (%s, %s, S/%s, S) attention matrix
    Memory: O(S^2/%s)  ->  %sx smaller!

  This is why CP is ESSENTIAL for long sequences (32K, 64K, 128K+).
  Without CP, attention memory explodes and you run out of GPU memory.
""",
            B,
            H,
            ws,
            B,
            H,
            ws,
            ws,
            ws,
        )
        logger.info("=" * 65)
        logger.info("  Results saved to outputs/cp_results.json")
        logger.info("  Run: python3 src/step3_plot.py")
        logger.info("=" * 65)
        logger.info("")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
