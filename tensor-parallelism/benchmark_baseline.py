"""Baseline benchmark -- single GPU, no tensor parallelism.

Run:
    python benchmark_baseline.py            # small model (29M params)
    python benchmark_baseline.py --large    # large model (~350M params, GQA)

Outputs results_no_tp.json for later comparison.
"""

import argparse
import json
import time

import torch
import torch.nn as nn

from model import (
    BATCH_SIZE,
    LARGE_CONFIG,
    NUM_BENCHMARK,
    NUM_WARMUP,
    SEQ_LEN,
    SMALL_CONFIG,
    StandardGPT,
    count_parameters,
    get_gpu_memory_mb,
    get_gpu_peak_memory_mb,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true",
                        help="Use LARGE_CONFIG (d=2048, 16 heads, GQA)")
    args = parser.parse_args()

    config = LARGE_CONFIG if args.large else SMALL_CONFIG
    tag = "large" if args.large else "small"

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    model = StandardGPT(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mem_model = get_gpu_memory_mb(device)
    n_params = count_parameters(model)

    torch.manual_seed(123)
    input_ids = torch.randint(
        0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device
    )
    labels = torch.randint(
        0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device
    )

    for _ in range(NUM_WARMUP):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    fwd_t, bwd_t, step_t = [], [], []
    for _ in range(NUM_BENCHMARK):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), labels.view(-1)
        )
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
    avg = lambda lst: sum(lst) / len(lst)

    results = dict(
        mode="no_tp",
        config=tag,
        num_gpus=1,
        total_params=n_params,
        params_per_gpu=n_params,
        mem_model_mb=round(mem_model, 2),
        mem_peak_mb=round(peak, 2),
        fwd_ms=round(avg(fwd_t) * 1000, 3),
        bwd_ms=round(avg(bwd_t) * 1000, 3),
        step_ms=round(avg(step_t) * 1000, 3),
        tokens_per_sec=round(BATCH_SIZE * SEQ_LEN / avg(step_t), 1),
        loss=round(loss.item(), 4),
    )

    with open("results_no_tp.json", "w") as f:
        json.dump(results, f, indent=2)

    sep = "=" * 60
    gqa_note = ""
    if config.n_kv_heads < config.n_heads:
        gqa_note = f" (GQA: {config.n_kv_heads} KV heads)"
    print(f"\n{sep}")
    print(f"  BASELINE (No TP) - 1 GPU - {tag}{gqa_note}")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}, d_ff={config.d_ff}")
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


if __name__ == "__main__":
    main()
