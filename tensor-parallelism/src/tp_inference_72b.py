"""Production TP inference with Qwen2.5-72B using PyTorch native TP.

Supports pure TP and TP+DP configurations in a single script.

Three deployment modes:
  1. Pure TP      -- torchrun --nproc_per_node=4 tp_inference_72b.py --tp-size 4
  2. Pure TP (2)  -- torchrun --nproc_per_node=2 tp_inference_72b.py --tp-size 2
  3. TP + DP      -- torchrun --nproc_per_node=8 tp_inference_72b.py --tp-size 4

When world_size == tp_size (pure TP, 1 replica):
  - TEST 1: batch=100 OOM stress test
  - TEST 2: TTFT (prefill latency) and generation throughput

When world_size > tp_size (TP+DP, multiple replicas):
  - TEST 3: multi-replica serving throughput (each replica processes
    diverse prompts independently, aggregate throughput is reported)

Uses torch.distributed.tensor.parallel (ColwiseParallel / RowwiseParallel)
with a 2D device mesh: (dp, tp). For pure TP the dp dimension is 1.

Requires: transformers, GPUs with enough combined VRAM for the model in bf16.
Outputs: results_tp{N}.json or results_tp{N}_dp{M}.json
"""

import argparse
import json
import logging
import os
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"

from rich.logging import RichHandler

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ================================================================
# Default constants
# ================================================================

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_PROMPTS_PER_REPLICA = 8

# Qwen2.5-72B architecture constants (for KV cache estimation)
N_LAYERS = 80
N_KV_HEADS = 8
D_HEAD = 128

BENCH_PROMPT = (
    "Explain the theory of general relativity and its implications for "
    "modern physics, including gravitational waves, black holes, and the "
    "expansion of the universe. Provide specific examples of experimental "
    "confirmations."
)

SERVING_PROMPTS = [
    "Explain the concept of tensor parallelism in distributed computing.",
    "What are the main differences between data parallelism and model parallelism?",
    "Describe how NVLink improves GPU-to-GPU communication bandwidth.",
    "What is the KV cache in transformer inference and why does it matter?",
    "How does grouped-query attention reduce memory usage in large models?",
    "Explain the SwiGLU activation function used in modern LLMs.",
    "What is the difference between prefill and decode phases in LLM serving?",
    "Describe how all-reduce works in a ring topology.",
]


# ================================================================
# KV cache memory estimation
# ================================================================


def kv_cache_memory_gb(
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
    tp_degree: int = 1,
) -> float:
    """Estimate KV cache memory per GPU in GB.

    KV cache stores key and value tensors for each layer. With GQA, only
    n_kv_heads (not n_heads) are cached. TP shards these across GPUs.
    """
    kv_heads_per_gpu = n_kv_heads // tp_degree if n_kv_heads >= tp_degree else 1
    total_bytes = (
        2 * n_layers * kv_heads_per_gpu * d_head * seq_len * batch_size * dtype_bytes
    )
    return total_bytes / 1e9


def log_kv_cache_table(
    tp_degree: int, mem_weights_gb: float, gpu_mem_gb: float, model_id: str
) -> None:
    """Log KV cache memory estimates for various batch/seq_len combos."""
    dash = "-" * 85
    logger.info("")
    logger.info(dash)
    logger.info("  KV Cache Memory -- %s (bf16, TP=%d)", model_id, tp_degree)
    logger.info(
        "  Weights/GPU: ~%.0f GB | GPU total: ~%.0f GB", mem_weights_gb, gpu_mem_gb
    )
    logger.info(dash)
    logger.info(
        "  %6s %8s | %10s %12s %8s", "Batch", "SeqLen", "KV/GPU", "Weights+KV", "Fits?"
    )
    logger.info("  %6s %8s | %10s %12s %8s", "---", "---", "---", "---", "---")

    for batch in [1, 8, 32, 100]:
        for seq_len in [2048, 8192]:
            kv = kv_cache_memory_gb(
                N_LAYERS, N_KV_HEADS, D_HEAD, seq_len, batch, 2, tp_degree
            )
            total = mem_weights_gb + kv
            fits = "OK" if total < gpu_mem_gb else "OOM!"
            logger.info(
                "  %6d %8d | %7.1fGB %9.1fGB %8s", batch, seq_len, kv, total, fits
            )
        logger.info("")
    logger.info(dash)


# ================================================================
# Model loading and TP application
# ================================================================


def load_and_parallelize(model_id: str, tp_mesh, device, global_rank: int):
    """Load model on CPU, apply TP, move to GPU. Returns (model, tokenizer)."""
    if global_rank == 0:
        logger.info("Loading model...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    if global_rank == 0:
        logger.info("CPU load: %.1fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    for layer in model.model.layers:
        parallelize_module(
            layer.self_attn,
            tp_mesh,
            {
                "q_proj": ColwiseParallel(),
                "k_proj": ColwiseParallel(),
                "v_proj": ColwiseParallel(),
                "o_proj": RowwiseParallel(),
            },
        )
        parallelize_module(
            layer.mlp,
            tp_mesh,
            {
                "gate_proj": ColwiseParallel(),
                "up_proj": ColwiseParallel(),
                "down_proj": RowwiseParallel(),
            },
        )
    model = model.to(device)
    if global_rank == 0:
        logger.info("TP + GPU transfer: %.1fs", time.perf_counter() - t0)

    return model, tokenizer


# ================================================================
# TEST 1: batch=100 OOM stress test (pure TP only)
# ================================================================


def test_batch100(model, tokenizer, device, tp_size, max_new_tokens):
    """Run batch=100 to prove TP solves the OOM that hits a single GPU."""
    rank = dist.get_rank()

    prompts_100 = [BENCH_PROMPT] * 100
    inputs_100 = tokenizer(
        prompts_100,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    if rank == 0:
        logger.info("")
        logger.info("--- TEST 1: Batch=100 (OOM'd on 1 GPU) ---")
        logger.info("Input shape: %s", list(inputs_100["input_ids"].shape))

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    dist.barrier()

    status, peak, tps, elapsed = "SKIPPED", 0.0, 0.0, 0.0

    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            out_100 = model.generate(
                **inputs_100, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        n_gen = out_100.shape[1] - inputs_100["input_ids"].shape[1]
        tps = (n_gen * 100) / elapsed
        status = "SUCCESS"

        if rank == 0:
            logger.info("SUCCESS -- generated %d tokens/sample", n_gen)
            logger.info("  %-25s %10.1f GB", "Peak memory/GPU", peak)
            logger.info("  %-25s %10.1f s", "Time", elapsed)
            logger.info("  %-25s %10.1f tok/s", "Throughput", tps)

        del out_100
    except torch.cuda.OutOfMemoryError:
        status = "OOM"
        if rank == 0:
            logger.warning("OOM at batch=100 even with TP=%d", tp_size)

    del inputs_100
    torch.cuda.empty_cache()

    return {
        "batch100_status": status,
        "batch100_peak_gb": round(peak, 1),
        "batch100_tps": round(tps, 1),
        "batch100_time_s": round(elapsed, 1),
    }


# ================================================================
# TEST 2: TTFT and throughput benchmark (pure TP only)
# ================================================================


def test_latency_throughput(model, tokenizer, device, max_new_tokens):
    """Measure prefill latency (TTFT) and generation throughput."""
    rank = dist.get_rank()
    BENCH_BATCH = 8

    prompts_bench = [BENCH_PROMPT] * BENCH_BATCH
    inputs_bench = tokenizer(
        prompts_bench,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    seq_len = inputs_bench["input_ids"].shape[1]

    with torch.no_grad():
        _ = model.generate(**inputs_bench, max_new_tokens=1)
    torch.cuda.synchronize()
    dist.barrier()

    # TTFT (prefill latency) -- 5 runs, drop first
    ttft_times = []
    for _ in range(5):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs_bench)
        torch.cuda.synchronize()
        ttft_times.append(time.perf_counter() - t0)
    avg_ttft = sum(ttft_times[1:]) / len(ttft_times[1:])

    # Generation throughput -- 3 runs
    torch.cuda.reset_peak_memory_stats(device)
    gen_times = []
    for _ in range(3):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs_bench, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        gen_times.append(time.perf_counter() - t0)

    avg_gen = sum(gen_times) / len(gen_times)
    n_generated = output_ids.shape[1] - inputs_bench["input_ids"].shape[1]
    tokens_per_sec = (n_generated * BENCH_BATCH) / avg_gen
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    if rank == 0:
        logger.info("")
        logger.info("--- TEST 2: Latency & Throughput (batch=%d) ---", BENCH_BATCH)
        logger.info("  %-25s %10.1f ms", "TTFT (prefill)", avg_ttft * 1000)
        logger.info(
            "  %-25s %10.0f ms  (%d tok/sample)",
            "Generation time",
            avg_gen * 1000,
            n_generated,
        )
        logger.info("  %-25s %10.1f tok/s", "Tokens/sec (total)", tokens_per_sec)
        logger.info("  %-25s %10.1f GB", "Peak memory/GPU", peak_mem)

    return {
        "bench_batch_size": BENCH_BATCH,
        "seq_len": seq_len,
        "ttft_ms": round(avg_ttft * 1000, 1),
        "gen_ms": round(avg_gen * 1000, 1),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "n_generated": n_generated,
        "mem_peak_gb": round(peak_mem, 1),
    }


# ================================================================
# TEST 3: TP+DP multi-replica serving throughput
# ================================================================


def test_dp_serving(
    model, tokenizer, device, tp_mesh, dp_mesh, max_new_tokens, prompts_per_replica
):
    """Each DP replica processes diverse prompts; aggregate throughput."""
    global_rank = dist.get_rank()
    tp_rank = tp_mesh.get_local_rank()
    dp_rank = dp_mesh.get_local_rank()
    tp_size = tp_mesh.size()
    n_replicas = dp_mesh.size()
    replica_id = dp_rank

    warmup_input = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**warmup_input, max_new_tokens=1)
    torch.cuda.synchronize()
    dist.barrier()

    if global_rank == 0:
        logger.info("")
        logger.info(
            "--- TEST 3: TP+DP Serving (%d replicas x TP=%d) ---", n_replicas, tp_size
        )

    torch.cuda.synchronize()
    dist.barrier()
    t_start = time.perf_counter()

    total_tokens = 0
    for i in range(prompts_per_replica):
        prompt = SERVING_PROMPTS[i % len(SERVING_PROMPTS)]
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        n_new = output_ids.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += n_new

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t_start
    replica_tps = total_tokens / elapsed

    if tp_rank == 0:
        logger.info(
            "Replica %d (GPUs %d-%d): %d tokens in %.2fs = %.1f tok/s",
            replica_id,
            global_rank,
            global_rank + tp_size - 1,
            total_tokens,
            elapsed,
            replica_tps,
        )

    # Only tp_rank==0 contributes to avoid double-counting across TP group
    tps_tensor = torch.tensor([replica_tps if tp_rank == 0 else 0.0], device=device)
    dist.all_reduce(tps_tensor, op=dist.ReduceOp.SUM)
    aggregate_tps = tps_tensor.item()

    if global_rank == 0:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  AGGREGATE THROUGHPUT")
        logger.info("=" * 60)
        logger.info("  %-25s %d replicas x TP=%d", "Config", n_replicas, tp_size)
        logger.info("  %-25s %d", "Total prompts", prompts_per_replica * n_replicas)
        logger.info("  %-25s %.2f s", "Wall time", elapsed)
        logger.info("  %-25s %.1f tok/s", "Aggregate tok/s", aggregate_tps)
        logger.info(
            "  %-25s %.1f tok/s", "Per-replica tok/s", aggregate_tps / n_replicas
        )
        logger.info("=" * 60)

    return {
        "n_replicas": n_replicas,
        "total_prompts": prompts_per_replica * n_replicas,
        "wall_time_s": round(elapsed, 2),
        "aggregate_tokens_per_sec": round(aggregate_tps, 1),
        "per_replica_tokens_per_sec": round(aggregate_tps / n_replicas, 1),
    }


# ================================================================
# CLI and main
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="TP (and TP+DP) inference benchmark for large models",
    )
    p.add_argument(
        "--tp-size",
        type=int,
        required=True,
        help="tensor parallelism degree (must divide world_size)",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument(
        "--prompts-per-replica",
        type=int,
        default=DEFAULT_PROMPTS_PER_REPLICA,
        help="prompts per replica in TP+DP mode (default: 8)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="directory for result JSON files (default: current directory)",
    )
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    tp_size = args.tp_size
    assert (
        world_size % tp_size == 0
    ), f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
    n_replicas = world_size // tp_size

    # 2D mesh: (dp, tp) -- for pure TP, dp dimension is 1
    mesh = init_device_mesh("cuda", (n_replicas, tp_size), mesh_dim_names=("dp", "tp"))
    tp_mesh = mesh["tp"]
    dp_mesh = mesh["dp"]

    if global_rank == 0:
        logger.info("=" * 60)
        if n_replicas == 1:
            logger.info("  %s", args.model_id)
            logger.info("  Pure TP=%d inference (%d GPUs)", tp_size, world_size)
        else:
            logger.info("  %s", args.model_id)
            logger.info(
                "  TP=%d x DP=%d (%d GPUs = %d replicas)",
                tp_size,
                n_replicas,
                world_size,
                n_replicas,
            )
        logger.info("=" * 60)

    model, tokenizer = load_and_parallelize(args.model_id, tp_mesh, device, global_rank)

    mem_weights = torch.cuda.max_memory_allocated(device) / 1e9
    gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    mem_free = gpu_total_gb - torch.cuda.memory_allocated(device) / 1e9

    if global_rank == 0:
        logger.info("  %-25s %10.1f GB", "Weights/GPU", mem_weights)
        logger.info("  %-25s %10.1f GB", "Free/GPU", mem_free)
        log_kv_cache_table(tp_size, mem_weights, gpu_total_gb, args.model_id)

    results = {
        "model_id": args.model_id,
        "tp_size": tp_size,
        "num_gpus": world_size,
        "mem_weights_gb": round(mem_weights, 1),
    }

    if n_replicas == 1:
        batch100_results = test_batch100(
            model, tokenizer, device, tp_size, args.max_new_tokens
        )
        results.update(batch100_results)

        latency_results = test_latency_throughput(
            model, tokenizer, device, args.max_new_tokens
        )
        results.update(latency_results)

        results["mode"] = f"tp{tp_size}"
        output_file = f"results_tp{tp_size}.json"
    else:
        serving_results = test_dp_serving(
            model,
            tokenizer,
            device,
            tp_mesh,
            dp_mesh,
            args.max_new_tokens,
            args.prompts_per_replica,
        )
        results.update(serving_results)

        results["mode"] = f"tp{tp_size}_dp{n_replicas}"
        output_file = f"results_tp{tp_size}_dp{n_replicas}.json"

    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, output_file)
        with open(output_path, "w") as fout:
            json.dump(results, fout, indent=2)
        logger.info("Results saved to %s", output_path)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
