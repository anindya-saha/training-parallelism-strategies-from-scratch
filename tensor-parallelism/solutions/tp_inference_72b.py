"""Production TP inference with Qwen2.5-72B using PyTorch native TP.

Demonstrates three things:
  1. TP solves OOM -- batch=100 that fails on 1 GPU works with TP=4
  2. TP reduces latency -- each GPU computes a smaller GEMM
  3. Measures TTFT (prefill) and generation throughput

Uses torch.distributed.tensor.parallel (ColwiseParallel / RowwiseParallel)
instead of from-scratch TP layers.

Run:
    torchrun --nproc_per_node=4 tp_inference_72b.py   # TP=4
    torchrun --nproc_per_node=2 tp_inference_72b.py   # TP=2 (for comparison)

Requires: transformers, 2+ GPUs with enough combined VRAM for 72B in bf16.
Outputs: results_tp{N}.json
"""

import json
import os
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
MAX_NEW_TOKENS = 128

PROMPT = (
    "Explain the theory of general relativity and its implications for "
    "modern physics, including gravitational waves, black holes, and the "
    "expansion of the universe. Provide specific examples of experimental "
    "confirmations."
)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  72B Inference with TP={world_size}")
        print(sep)

    # --- Device mesh ---
    tp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

    # --- Load model ---
    if rank == 0:
        print("  Loading model...", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    if rank == 0:
        print(f"  CPU load: {time.perf_counter() - t0:.1f}s", flush=True)

    # --- Apply TP (Megatron-LM pattern) ---
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
    if rank == 0:
        print(f"  TP + GPU transfer: {time.perf_counter() - t0:.1f}s", flush=True)

    mem_weights = torch.cuda.max_memory_allocated(device) / 1e9
    mem_free = (
        torch.cuda.get_device_properties(device).total_memory
        - torch.cuda.memory_allocated(device)
    ) / 1e9
    if rank == 0:
        print(f"  Weights/GPU: {mem_weights:.1f} GB | Free/GPU: {mem_free:.1f} GB")

    # ================================================================
    # TEST 1: batch=100 -- the workload that OOM'd on 1 GPU
    # ================================================================
    batch100_status = "SKIPPED"
    batch100_peak = 0.0
    batch100_tps = 0.0
    batch100_time = 0.0

    prompts_100 = [PROMPT] * 100
    inputs_100 = tokenizer(
        prompts_100,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    if rank == 0:
        print()
        print("  --- Batch=100 Test (OOM'd on 1 GPU) ---")
        print(f"  Input shape: {inputs_100['input_ids'].shape}")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    dist.barrier()

    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            out_100 = model.generate(
                **inputs_100, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        torch.cuda.synchronize()
        batch100_time = time.perf_counter() - t0
        batch100_peak = torch.cuda.max_memory_allocated(device) / 1e9
        n_gen = out_100.shape[1] - inputs_100["input_ids"].shape[1]
        batch100_tps = (n_gen * 100) / batch100_time
        batch100_status = "SUCCESS"

        if rank == 0:
            print(f"  SUCCESS! Generated {n_gen} tokens/sample")
            print(f"  Peak memory/GPU: {batch100_peak:.1f} GB")
            print(f"  Time: {batch100_time:.1f}s")
            print(f"  Throughput: {batch100_tps:.1f} tok/s")

        del out_100
    except torch.cuda.OutOfMemoryError:
        batch100_status = "OOM"
        if rank == 0:
            print(f"  OOM at batch=100 even with TP={world_size}")

    del inputs_100
    torch.cuda.empty_cache()

    # ================================================================
    # TEST 2: TTFT & throughput benchmark (batch=8)
    # ================================================================
    BENCH_BATCH = 8
    prompts_bench = [PROMPT] * BENCH_BATCH
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

    # TTFT (prefill latency)
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

    # Generation throughput
    torch.cuda.reset_peak_memory_stats(device)
    gen_times = []
    for _ in range(3):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs_bench, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        torch.cuda.synchronize()
        gen_times.append(time.perf_counter() - t0)

    avg_gen = sum(gen_times) / len(gen_times)
    n_generated = output_ids.shape[1] - inputs_bench["input_ids"].shape[1]
    tokens_per_sec = (n_generated * BENCH_BATCH) / avg_gen
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    if rank == 0:
        print()
        print(f"  --- Latency & Throughput (batch={BENCH_BATCH}) ---")
        print(f"  TTFT (prefill):     {avg_ttft * 1000:.1f} ms")
        print(
            f"  Generation time:    {avg_gen * 1000:.0f} ms "
            f"({n_generated} tok/sample)"
        )
        print(f"  Tokens/sec (total): {tokens_per_sec:.1f}")
        print(f"  Peak memory/GPU:    {peak_mem:.1f} GB")

        results = {
            "tp_size": world_size,
            "batch100_status": batch100_status,
            "batch100_peak_gb": round(batch100_peak, 1),
            "batch100_tps": round(batch100_tps, 1),
            "batch100_time_s": round(batch100_time, 1),
            "bench_batch_size": BENCH_BATCH,
            "seq_len": seq_len,
            "mem_weights_gb": round(mem_weights, 1),
            "mem_peak_gb": round(peak_mem, 1),
            "ttft_ms": round(avg_ttft * 1000, 1),
            "gen_ms": round(avg_gen * 1000, 1),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "n_generated": n_generated,
        }
        with open(f"results_tp{world_size}.json", "w") as fout:
            json.dump(results, fout, indent=2)
        print(f"\n  Results saved to results_tp{world_size}.json")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
