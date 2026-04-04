"""
TP Scaling Study -- measure how tensor parallelism scales on a real model.

Loads an open-source 3B model (default: meta-llama/Llama-3.2-3B), applies
HuggingFace native TP via tp_plan="auto", and measures:
  1. Throughput scaling (tokens/sec/GPU vs TP degree)
  2. Maximum batch size before OOM at each TP degree

Two modes:
  collect  -- run benchmarks at multiple TP degrees, save JSON results
  plot     -- read collected results and generate charts

Usage:
  # Collect at TP=1,2,4,8 (launches torchrun for each degree)
  python tp_scaling_study.py collect --tp-degrees 1 2 4 8

  # Plot from collected results
  python tp_scaling_study.py plot --results-dir results_scaling

  # Collect + plot in one shot
  python tp_scaling_study.py collect --tp-degrees 1 2 4 8 --plot

  # Custom model
  python tp_scaling_study.py collect --tp-degrees 1 2 4 8 \
      --model-id meta-llama/Llama-3.2-3B --plot
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B"
DEFAULT_SEQ_LEN = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_WARMUP = 3
DEFAULT_TRIALS = 10
DEFAULT_DTYPE = "bfloat16"


# ================================================================
# Single-run worker (called via torchrun)
# ================================================================


def run_worker(args) -> None:
    """Run inside a torchrun process. Measures throughput and max batch size."""
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoConfig

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    if args.tp_degree > 1:
        dist.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(device)

    rank = 0 if args.tp_degree == 1 else dist.get_rank()
    ws = args.tp_degree
    dtype = getattr(torch, args.dtype)

    if rank == 0:
        logger.info("Loading %s (TP=%d, dtype=%s)", args.model_id, ws, args.dtype)

    tp_plan = "auto" if ws > 1 else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        tp_plan=tp_plan,
    )
    if ws == 1:
        model = model.to(device)
    model.eval()

    config = AutoConfig.from_pretrained(args.model_id)
    vocab_size = config.vocab_size

    n_params_local = sum(p.numel() for p in model.parameters())

    if rank == 0:
        logger.info("Params per GPU: %s", f"{n_params_local:,}")

    # -- Throughput measurement --
    torch.cuda.reset_peak_memory_stats(device)

    input_ids = torch.randint(
        0, vocab_size, (args.batch_size, args.seq_len), device=device
    )

    if rank == 0:
        logger.info(
            "Throughput benchmark: batch_size=%d, seq_len=%d",
            args.batch_size,
            args.seq_len,
        )

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(input_ids)
        torch.cuda.synchronize()

        times = []
        for _ in range(args.trials):
            torch.cuda.synchronize()
            if ws > 1:
                dist.barrier()
            t0 = time.perf_counter()
            _ = model(input_ids)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    avg_time = sum(times) / len(times)
    tokens_per_sec = args.batch_size * args.seq_len / avg_time
    mem_peak = torch.cuda.max_memory_allocated(device) / 1e6

    if rank == 0:
        logger.info(
            "Avg forward: %.2f ms | Tokens/sec: %.0f | Peak mem: %.0f MB",
            avg_time * 1000,
            tokens_per_sec,
            mem_peak,
        )

    # -- Max batch size search (binary search) --
    if rank == 0:
        logger.info("Searching for max batch size (seq_len=%d)...", args.seq_len)

    lo, hi = 1, args.max_batch_probe
    best = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        success = _try_batch_size(model, mid, args.seq_len, vocab_size, device, ws)

        if ws > 1:
            success_tensor = torch.tensor([1 if success else 0], device=device)
            dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)
            success = success_tensor.item() == 1

        if success:
            best = mid
            lo = mid + 1
            if rank == 0:
                logger.info("  batch_size=%d -> OK", mid)
        else:
            hi = mid - 1
            if rank == 0:
                logger.info("  batch_size=%d -> OOM", mid)

    if rank == 0:
        logger.info("Max batch size: %d", best)

    # -- Save results --
    if rank == 0:
        results = dict(
            model_id=args.model_id,
            tp_degree=ws,
            dtype=args.dtype,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            params_per_gpu=n_params_local,
            avg_forward_ms=round(avg_time * 1000, 2),
            tokens_per_sec=round(tokens_per_sec, 1),
            tokens_per_sec_per_gpu=round(tokens_per_sec / ws, 1),
            mem_peak_mb=round(mem_peak, 1),
            max_batch_size=best,
        )

        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", output_path)

    if ws > 1:
        dist.destroy_process_group()


def _try_batch_size(
    model, batch_size: int, seq_len: int, vocab_size: int, device: torch.device, ws: int
) -> bool:
    """Try a forward pass at the given batch size. Returns True if it fits."""
    import torch

    try:
        torch.cuda.empty_cache()
        gc.collect()
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        del x
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return False


# ================================================================
# Collect mode (orchestrator)
# ================================================================


def collect(args) -> None:
    """Launch torchrun for each TP degree and collect results."""
    script_path = Path(__file__).resolve()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for tp in args.tp_degrees:
        output_file = results_dir / f"tp_{tp}.json"

        worker_args = [
            "--mode",
            "worker",
            "--model-id",
            args.model_id,
            "--tp-degree",
            str(tp),
            "--seq-len",
            str(args.seq_len),
            "--batch-size",
            str(args.batch_size),
            "--warmup",
            str(args.warmup),
            "--trials",
            str(args.trials),
            "--dtype",
            args.dtype,
            "--max-batch-probe",
            str(args.max_batch_probe),
            "--output-file",
            str(output_file),
        ]

        if tp == 1:
            cmd = [sys.executable, str(script_path)] + worker_args
        else:
            cmd = [
                "torchrun",
                f"--nproc_per_node={tp}",
                str(script_path),
            ] + worker_args

        logger.info("=" * 60)
        logger.info("Launching TP=%d", tp)
        logger.info("=" * 60)

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            logger.error("TP=%d FAILED (exit %d)", tp, result.returncode)
        else:
            logger.info("TP=%d complete", tp)

    logger.info("All runs complete. Results in %s/", results_dir)

    if args.plot:
        args.results_dir = str(results_dir)
        plot(args)


# ================================================================
# Plot mode
# ================================================================


def load_scaling_results(results_dir: str) -> list[dict]:
    """Load all tp_*.json files, sorted by TP degree."""
    results = []
    for path in sorted(Path(results_dir).glob("tp_*.json")):
        with open(path) as f:
            results.append(json.load(f))
    results.sort(key=lambda r: r["tp_degree"])
    return results


def plot(args) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1, palette="muted")
    palette = sns.color_palette("muted")
    drop_color = sns.color_palette("pastel")[3]

    results = load_scaling_results(args.results_dir)
    if len(results) < 2:
        logger.error("Need at least 2 TP results to plot (found %d)", len(results))
        sys.exit(1)

    tp_degrees = [r["tp_degree"] for r in results]
    tp_labels = [str(t) for t in tp_degrees]
    model_id = results[0].get("model_id", "Model")
    model_short = model_id.split("/")[-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"{model_short}: TP Scaling Study", fontsize=16, fontweight="bold", y=1.02
    )

    # -- 1. Throughput scaling (tokens/sec/GPU) with performance drop --
    ax = axes[0]
    tps_per_gpu = [r["tokens_per_sec_per_gpu"] for r in results]
    baseline_tps = tps_per_gpu[0]
    pct_drop = [(1 - t / baseline_tps) * 100 for t in tps_per_gpu]

    bars = ax.bar(tp_labels, tps_per_gpu, color=palette[0], zorder=3)
    ax.set_title("Throughput Scaling with TP", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tensor Parallelism (TP)")
    ax.set_ylabel("Tokens/sec/GPU")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e3:.1f}K" if x >= 1000 else f"{x:.0f}")
    )

    for bar, tps, drop in zip(bars, tps_per_gpu, pct_drop):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tps_per_gpu) * 0.02,
            f"{tps:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        if drop > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.5,
                f"-{drop:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="#d62728",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="#d62728",
                ),
            )

    # -- 2. Max batch size per TP degree --
    ax = axes[1]
    max_bs = [r["max_batch_size"] for r in results]

    bars = ax.bar(tp_labels, max_bs, color=palette[2], zorder=3)
    ax.set_title("Maximum Batch Size per TP Value", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tensor Parallelism (TP)")
    ax.set_ylabel("Maximum Batch Size")

    for bar, bs in zip(bars, max_bs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(max_bs) * 0.02,
            str(bs),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=palette[2],
        )

    sns.despine(fig=fig)
    plt.tight_layout()

    output_path = args.plot_output or f"tp_scaling_{model_short}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info("Plot saved to %s", output_path)
    plt.show()


# ================================================================
# CLI
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="TP scaling study on real open-source models",
    )

    p.add_argument("--mode", choices=["worker"], default=None, help=argparse.SUPPRESS)

    sub = p.add_subparsers(dest="command")

    # -- collect --
    c = sub.add_parser("collect", help="run benchmarks at multiple TP degrees")
    c.add_argument(
        "--tp-degrees",
        type=int,
        nargs="+",
        required=True,
        help="TP degrees to benchmark (e.g. 1 2 4 8)",
    )
    c.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    c.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    c.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="batch size for throughput measurement",
    )
    c.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    c.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    c.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["float16", "bfloat16", "float32"],
    )
    c.add_argument(
        "--max-batch-probe",
        type=int,
        default=128,
        help="upper bound for max batch size binary search",
    )
    c.add_argument(
        "--results-dir",
        type=str,
        default="results_scaling",
        help="directory to save results",
    )
    c.add_argument("--plot", action="store_true", default=False)
    c.add_argument("--plot-output", type=str, default=None)

    # -- plot --
    pl = sub.add_parser("plot", help="generate charts from collected results")
    pl.add_argument("--results-dir", type=str, required=True)
    pl.add_argument("--plot-output", type=str, default=None)

    # -- worker args (hidden, used internally by torchrun) --
    p.add_argument(
        "--model-id", type=str, default=DEFAULT_MODEL_ID, dest="model_id_worker"
    )
    p.add_argument("--tp-degree", type=int, default=1)
    p.add_argument(
        "--seq-len", type=int, default=DEFAULT_SEQ_LEN, dest="seq_len_worker"
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, dest="batch_size_worker"
    )
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, dest="warmup_worker")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS, dest="trials_worker")
    p.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, dest="dtype_worker")
    p.add_argument(
        "--max-batch-probe", type=int, default=128, dest="max_batch_probe_worker"
    )
    p.add_argument("--output-file", type=str, default="result.json")

    return p.parse_args()


def main():
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    args = parse_args()

    if args.mode == "worker":
        worker_args = argparse.Namespace(
            model_id=args.model_id_worker,
            tp_degree=args.tp_degree,
            seq_len=args.seq_len_worker,
            batch_size=args.batch_size_worker,
            warmup=args.warmup_worker,
            trials=args.trials_worker,
            dtype=args.dtype_worker,
            max_batch_probe=args.max_batch_probe_worker,
            output_file=args.output_file,
        )
        run_worker(worker_args)
    elif args.command == "collect":
        collect(args)
    elif args.command == "plot":
        plot(args)
    else:
        logger.error("Specify a command: collect or plot")
        sys.exit(1)


if __name__ == "__main__":
    main()
