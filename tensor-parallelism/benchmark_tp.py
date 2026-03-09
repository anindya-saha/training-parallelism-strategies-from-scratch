"""Tensor Parallelism benchmark -- multi-GPU.

Run:
    torchrun --nproc_per_node=N benchmark_tp.py               # small, TP only
    torchrun --nproc_per_node=N benchmark_tp.py --large       # large (GQA)
    torchrun --nproc_per_node=N benchmark_tp.py --sp          # small + SP
    torchrun --nproc_per_node=N benchmark_tp.py --large --sp  # large (GQA) + SP

where N must divide n_heads and n_kv_heads evenly.
Outputs results_tp.json for later comparison.
"""

import argparse
import json
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from model import (
    BATCH_SIZE,
    LARGE_CONFIG,
    NUM_BENCHMARK,
    NUM_WARMUP,
    SEQ_LEN,
    SMALL_CONFIG,
    count_parameters,
    get_gpu_memory_mb,
    get_gpu_peak_memory_mb,
)
from tp import TPGPT, tp_cross_entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true",
                        help="Use LARGE_CONFIG (d=2048, 16 heads, GQA)")
    parser.add_argument("--sp", action="store_true",
                        help="Enable sequence parallelism")
    args = parser.parse_args()

    config = LARGE_CONFIG if args.large else SMALL_CONFIG
    tag = "large" if args.large else "small"
    sp_tag = "+SP" if args.sp else ""

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    if rank == 0:
        gqa_note = ""
        if config.n_kv_heads < config.n_heads:
            gqa_note = f", GQA({config.n_kv_heads} KV heads)"
        print(f"\nTP{sp_tag} benchmark: {ws} GPUs, {tag}{gqa_note}", flush=True)
        print(
            f"  d_model={config.d_model}, n_heads={config.n_heads}, "
            f"d_ff={config.d_ff}, n_layers={config.n_layers}",
            flush=True,
        )
        heads_local = config.n_heads // ws
        kv_local = config.n_kv_heads // ws
        print(
            f"  Each GPU: {heads_local}/{config.n_heads} Q heads, "
            f"{kv_local}/{config.n_kv_heads} KV heads, "
            f"{config.d_ff // ws}/{config.d_ff} FFN dim",
            flush=True,
        )

    model = TPGPT(config, sequence_parallel=args.sp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mem_model = get_gpu_memory_mb(device)
    n_local = count_parameters(model)

    torch.manual_seed(123)
    input_ids = torch.randint(
        0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device
    )
    labels = torch.randint(
        0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device
    )

    for _ in range(NUM_WARMUP):
        logits = model(input_ids)
        loss = tp_cross_entropy(logits, labels, config.vocab_size)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    dist.barrier()

    fwd_t, bwd_t, step_t = [], [], []
    for _ in range(NUM_BENCHMARK):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()

        logits = model(input_ids)
        loss = tp_cross_entropy(logits, labels, config.vocab_size)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        fwd_t.append(t1 - t0)
        bwd_t.append(t2 - t1)
        step_t.append(t3 - t0)

    peak = get_gpu_peak_memory_mb(device)

    for r in range(ws):
        if rank == r:
            print(
                f"  [GPU {rank}] params: {n_local:,} | "
                f"model: {mem_model:.1f} MB | peak: {peak:.1f} MB",
                flush=True,
            )
        dist.barrier()

    if rank == 0:
        avg = lambda lst: sum(lst) / len(lst)
        results = dict(
            mode=f"tp_{ws}{'_sp' if args.sp else ''}",
            config=tag,
            num_gpus=ws,
            sequence_parallel=args.sp,
            params_per_gpu=n_local,
            mem_model_mb=round(mem_model, 2),
            mem_peak_mb=round(peak, 2),
            fwd_ms=round(avg(fwd_t) * 1000, 3),
            bwd_ms=round(avg(bwd_t) * 1000, 3),
            step_ms=round(avg(step_t) * 1000, 3),
            tokens_per_sec=round(BATCH_SIZE * SEQ_LEN / avg(step_t), 1),
            loss=round(loss.item(), 4),
        )
        with open("results_tp.json", "w") as f:
            json.dump(results, f, indent=2)

        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  TENSOR PARALLELISM{sp_tag} - {ws} GPUs - {tag}")
        print(sep)
        for k in [
            "params_per_gpu",
            "mem_model_mb",
            "mem_peak_mb",
            "fwd_ms",
            "bwd_ms",
            "step_ms",
            "tokens_per_sec",
            "loss",
        ]:
            v = results[k]
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                print(f"  {label:<25} {v:>12.2f}")
            else:
                print(f"  {label:<25} {v:>12,}")
        print(sep)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
