#!/usr/bin/env python3
"""
Baseline: Single GPU training (no parallelism)
Uses PyTorch AMP (FP16) for fair comparison with DeepSpeed.
"""

import os, time, json, argparse
import torch
import torch.nn as nn
from gpt2_model import (
    get_sequential_model, lm_loss_fn, count_parameters,
    VOCAB_SIZE, MAX_SEQ_LEN
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    print(f"\n{'='*60}")
    print(f"  BASELINE: Single GPU — batch_size={args.batch_size}")
    print(f"{'='*60}")

    # Build model (FP32 weights — autocast handles FP16 in forward pass)
    model = get_sequential_model().to(device)
    n_params = count_parameters(model)
    print(f"  Model: {n_params/1e9:.2f}B params on {torch.cuda.get_device_name(0)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")

    # Synthetic data
    def get_batch():
        tokens = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.seq_len + 1), device=device)
        return tokens[:, :-1], tokens[:, 1:]

    # Training loop
    step_times = []
    for step in range(args.warmup_steps + args.num_steps):
        input_ids, labels = get_batch()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(input_ids)
            loss = lm_loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if step >= args.warmup_steps:
            step_times.append(t1 - t0)

        if step % 5 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f} | "
                  f"Time: {(t1-t0)*1000:.1f}ms | "
                  f"Mem: {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB")

    # Results
    avg_time = sum(step_times) / len(step_times)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    throughput = args.batch_size * args.seq_len / avg_time  # tokens/sec

    results = {
        "experiment": "baseline_1gpu",
        "batch_size": args.batch_size,
        "avg_step_time_ms": avg_time * 1000,
        "throughput_tokens_per_sec": throughput,
        "peak_gpu_memory_gb": peak_mem,
        "step_times_ms": [t * 1000 for t in step_times],
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS: Baseline (1 GPU)")
    print(f"  Avg step time:   {avg_time*1000:.1f} ms")
    print(f"  Throughput:      {throughput:,.0f} tokens/sec")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")
    print(f"{'='*60}\n")

    with open("results_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results_baseline.json")


if __name__ == "__main__":
    main()
