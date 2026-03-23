"""Side-by-side comparison of baseline vs TP results.

Run the two benchmarks first:
    python benchmark_baseline.py
    torchrun --nproc_per_node=N benchmark_tp.py

Then:
    python compare_results.py
"""

import json
import sys

from model import N_LAYERS


def main():
    try:
        with open("results_no_tp.json") as f:
            no_tp = json.load(f)
        with open("results_tp.json") as f:
            tp = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run benchmark_baseline.py and benchmark_tp.py first.")
        sys.exit(1)

    ng = tp["num_gpus"]
    p1, p2 = no_tp["params_per_gpu"], tp["params_per_gpu"]
    m1, m2 = no_tp["mem_model_mb"], tp["mem_model_mb"]
    pk1, pk2 = no_tp["mem_peak_mb"], tp["mem_peak_mb"]
    speedup = no_tp["step_ms"] / tp["step_ms"]
    eff = speedup / ng * 100

    sep = "=" * 74
    dash = "-" * 68

    print(f"\n{sep}")
    print("  TENSOR PARALLELISM: SIDE-BY-SIDE COMPARISON")
    print(sep)

    header_r = f"TP ({ng} GPUs)"
    print(f"  {'Metric':<28} {'No TP (1 GPU)':>20} {header_r:>20}")

    rows = [
        ("Parameters/GPU", f"{p1:,}", f"{p2:,}"),
        ("  Reduction", "---", f"{(1 - p2 / p1) * 100:.1f}% fewer"),
        ("---", "", ""),
        ("Model memory (MB)", f"{m1:.1f}", f"{m2:.1f}"),
        ("  Reduction", "---", f"{(1 - m2 / m1) * 100:.1f}% less"),
        ("Peak memory (MB)", f"{pk1:.1f}", f"{pk2:.1f}"),
        ("  Reduction", "---", f"{(1 - pk2 / pk1) * 100:.1f}% less"),
        ("---", "", ""),
        ("Forward (ms)", f"{no_tp['fwd_ms']:.2f}", f"{tp['fwd_ms']:.2f}"),
        ("Backward (ms)", f"{no_tp['bwd_ms']:.2f}", f"{tp['bwd_ms']:.2f}"),
        ("Full step (ms)", f"{no_tp['step_ms']:.2f}", f"{tp['step_ms']:.2f}"),
        ("  Speedup", "---", f"{speedup:.2f}x"),
        ("  Ideal", "---", f"{ng:.0f}x"),
        ("  Efficiency", "---", f"{eff:.1f}%"),
        ("---", "", ""),
        (
            "Throughput (tok/s)",
            f"{no_tp['tokens_per_sec']:,.0f}",
            f"{tp['tokens_per_sec']:,.0f}",
        ),
        ("Loss", f"{no_tp['loss']:.4f}", f"{tp['loss']:.4f}"),
    ]

    for label, v1, v2 in rows:
        if label == "---":
            print("  " + dash)
        else:
            print(f"  {label:<28} {v1:>20} {v2:>20}")
    print(sep)

    print(f"\n  ANALYSIS:")
    print(
        f"  * Param reduction: {(1 - p2 / p1) * 100:.1f}% "
        f"(not 50% -- LayerNorm + embeddings replicated)"
    )
    print(
        f"  * Peak mem reduction: {(1 - pk2 / pk1) * 100:.1f}% "
        f"(LN activations still full d_model)"
    )
    print(f"  * Speedup: {speedup:.2f}x / {ng}x ideal = {eff:.1f}% efficiency")
    n_allreduce = N_LAYERS * 2
    print(
        f"  * Comm overhead: ~{100 - eff:.1f}% "
        f"({N_LAYERS} layers x 2 all-reduces = {n_allreduce}/step)"
    )


if __name__ == "__main__":
    main()
