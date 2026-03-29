import argparse
import json
import logging
import sys

from rich.logging import RichHandler

logger = logging.getLogger(__name__)

RESULT_FILES = {
    "gpt": ("results_model_gpt.json", "results_model_gpt_tp.json"),
    "llama": ("results_model_llama.json", "results_model_llama_tp.json"),
}

COLORS_FALLBACK = ["#4C72B0", "#DD8452"]


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pct_reduction(baseline: float, tp: float) -> str:
    return f"{(1 - tp / baseline) * 100:.1f}%"


def compare(baseline: dict, tp: dict, model_name: str) -> None:
    ng = tp["num_gpus"]
    n_layers = tp.get("n_layers", baseline.get("n_layers", "?"))

    p1, p2 = baseline["params_per_gpu"], tp["params_per_gpu"]
    m1, m2 = baseline["mem_model_mb"], tp["mem_model_mb"]
    pk1, pk2 = baseline["mem_peak_mb"], tp["mem_peak_mb"]
    speedup = baseline["step_ms"] / tp["step_ms"]
    efficiency = speedup / ng * 100

    sep = "=" * 74
    dash = "-" * 68

    logger.info("")
    logger.info(sep)
    logger.info("  %s: TENSOR PARALLELISM COMPARISON", model_name.upper())
    logger.info(sep)
    logger.info("  %-28s %20s %20s", "Metric", "No TP (1 GPU)", f"TP ({ng} GPUs)")

    rows = [
        ("Parameters/GPU", f"{p1:,}", f"{p2:,}"),
        ("  Reduction", "---", f"{pct_reduction(p1, p2)} fewer"),
        None,
        ("Model memory (MB)", f"{m1:.1f}", f"{m2:.1f}"),
        ("  Reduction", "---", f"{pct_reduction(m1, m2)} less"),
        ("Peak memory (MB)", f"{pk1:.1f}", f"{pk2:.1f}"),
        ("  Reduction", "---", f"{pct_reduction(pk1, pk2)} less"),
        None,
        ("Forward (ms)", f"{baseline['fwd_ms']:.2f}", f"{tp['fwd_ms']:.2f}"),
        ("Backward (ms)", f"{baseline['bwd_ms']:.2f}", f"{tp['bwd_ms']:.2f}"),
        ("Full step (ms)", f"{baseline['step_ms']:.2f}", f"{tp['step_ms']:.2f}"),
        ("  Speedup", "---", f"{speedup:.2f}x"),
        ("  Ideal", "---", f"{ng:.0f}x"),
        ("  Efficiency", "---", f"{efficiency:.1f}%"),
        None,
        ("Throughput (tok/s)", f"{baseline['tokens_per_sec']:,.0f}",
         f"{tp['tokens_per_sec']:,.0f}"),
        ("Loss", f"{baseline['loss']:.4f}", f"{tp['loss']:.4f}"),
    ]

    for row in rows:
        if row is None:
            logger.info("  %s", dash)
        else:
            label, v1, v2 = row
            logger.info("  %-28s %20s %20s", label, v1, v2)

    logger.info(sep)
    logger.info("")
    logger.info("  ANALYSIS:")
    logger.info(
        "  * Param reduction: %s (not 50%% -- LayerNorm + embeddings replicated)",
        pct_reduction(p1, p2),
    )
    logger.info(
        "  * Peak mem reduction: %s (LN activations still full d_model)",
        pct_reduction(pk1, pk2),
    )
    logger.info(
        "  * Speedup: %.2fx / %dx ideal = %.1f%% efficiency",
        speedup, ng, efficiency,
    )
    logger.info(
        "  * Comm overhead: ~%.1f%% (%s layers x 2 all-reduces = %s/step)",
        100 - efficiency, n_layers, f"{n_layers * 2}" if isinstance(n_layers, int) else "?",
    )
    logger.info("")


# ================================================================
# Visualization (--plot)
# ================================================================


def _bar_with_label(ax, labels, vals, fmt=None, colors=None, palette=None):
    """Draw a bar chart and place value labels above each bar."""
    if fmt is None:
        fmt = lambda v: f"{v:.1f}"
    elif isinstance(fmt, str):
        _s = fmt
        fmt = lambda v, _s=_s: _s % v
    colors = colors or palette or COLORS_FALLBACK
    bars = ax.bar(labels, vals, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    headroom = max(vals) * 0.02
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + headroom,
                fmt(v), ha="center", va="bottom", fontweight="bold", fontsize=11)
    return bars


def _badge(ax, text, x=0.95, y=0.92, color="#2ca02c", bg="#e6ffe6"):
    ax.text(x, y, text, transform=ax.transAxes, ha="right", fontsize=12,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, edgecolor=color))


def plot_comparison(baseline: dict, tp: dict, model_name: str,
                    output_path: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1, palette="muted")
    palette = sns.color_palette("muted", 2)
    ideal_color = sns.color_palette("pastel", 2)[0]

    ng = tp["num_gpus"]
    mode_labels = [f"No TP\n(1 GPU)", f"TP\n({ng} GPUs)"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{model_name.upper()}: Tensor Parallelism Comparison",
                 fontsize=18, fontweight="bold", y=0.98)

    # 1. Parameters per GPU
    ax = axes[0, 0]
    vals = [baseline["params_per_gpu"], tp["params_per_gpu"]]
    _bar_with_label(ax, mode_labels, vals,
                    fmt=lambda v: f"{v/1e6:.2f}M", palette=palette)
    ax.set_title("Parameters per GPU", fontsize=13, fontweight="bold")
    ax.set_ylabel("Parameter Count")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    reduction = (1 - tp["params_per_gpu"] / baseline["params_per_gpu"]) * 100
    _badge(ax, f"v {reduction:.1f}%")

    # 2. Model memory
    ax = axes[0, 1]
    vals = [baseline["mem_model_mb"], tp["mem_model_mb"]]
    _bar_with_label(ax, mode_labels, vals, fmt="%.1f", palette=palette)
    ax.set_title("Model Memory (MB)", fontsize=13, fontweight="bold")
    ax.set_ylabel("MB")
    reduction = (1 - tp["mem_model_mb"] / baseline["mem_model_mb"]) * 100
    _badge(ax, f"v {reduction:.1f}%")

    # 3. Peak memory
    ax = axes[0, 2]
    vals = [baseline["mem_peak_mb"], tp["mem_peak_mb"]]
    _bar_with_label(ax, mode_labels, vals, fmt="%.1f", palette=palette)
    ax.set_title("Peak Memory (MB)", fontsize=13, fontweight="bold")
    ax.set_ylabel("MB")
    reduction = (1 - tp["mem_peak_mb"] / baseline["mem_peak_mb"]) * 100
    _badge(ax, f"v {reduction:.1f}%")

    # 4. Latency breakdown
    ax = axes[1, 0]
    x = np.arange(3)
    width = 0.3
    metrics = ["fwd_ms", "bwd_ms", "step_ms"]
    metric_labels = ["Forward", "Backward", "Full Step"]
    v_no = [baseline[m] for m in metrics]
    v_tp = [tp[m] for m in metrics]
    bars1 = ax.bar(x - width / 2, v_no, width, label="No TP (1 GPU)",
                   color=palette[0])
    bars2 = ax.bar(x + width / 2, v_tp, width, label=f"TP ({ng} GPUs)",
                   color=palette[1])
    ax.set_title("Latency Breakdown (ms)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=10)
    headroom = max(v_no + v_tp) * 0.02
    for bar, v in zip(list(bars1) + list(bars2), v_no + v_tp):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + headroom,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 5. Throughput
    ax = axes[1, 1]
    vals = [baseline["tokens_per_sec"], tp["tokens_per_sec"]]
    _bar_with_label(ax, mode_labels, vals,
                    fmt=lambda v: f"{v:,.0f}", palette=palette)
    ax.set_title("Throughput (tokens/sec)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Tokens / sec")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e3:.1f}K" if x >= 1000 else f"{x:.0f}"))
    speedup_thr = tp["tokens_per_sec"] / baseline["tokens_per_sec"]
    _badge(ax, f"^ {speedup_thr:.2f}x")

    # 6. Speedup and efficiency
    ax = axes[1, 2]
    step_speedup = baseline["step_ms"] / tp["step_ms"]
    ideal = ng
    efficiency = step_speedup / ideal * 100
    bar_vals = [ideal, step_speedup]
    bar_labels = [f"Ideal\n({ng}x)", f"Actual\n({step_speedup:.2f}x)"]
    bar_colors = [ideal_color, palette[1]]
    bars = ax.bar(bar_labels, bar_vals, color=bar_colors, width=0.5)
    ax.set_title(f"Speedup ({ng}-GPU TP)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Speedup (x)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No speedup")
    headroom = max(bar_vals) * 0.02
    for bar, v in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + headroom,
                f"{v:.2f}x", ha="center", va="bottom", fontweight="bold", fontsize=12)
    _badge(ax, f"Efficiency: {efficiency:.1f}%", color="#d62728", bg="#ffe6e6")
    comm_overhead = 100 - efficiency
    _badge(ax, f"Comm overhead: ~{comm_overhead:.1f}%", y=0.80,
           color="#666666", bg="#f0f0f0")

    sns.despine(fig=fig)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info("Plot saved to %s", output_path)
    plt.show()


# ================================================================
# CLI
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare baseline vs TP benchmark results",
    )
    p.add_argument(
        "--model", choices=list(RESULT_FILES.keys()), default="gpt",
        help="which model pair to compare (default: gpt)",
    )
    p.add_argument(
        "--baseline", type=str, default=None,
        help="override path to baseline results JSON",
    )
    p.add_argument(
        "--tp", type=str, default=None,
        help="override path to TP results JSON",
    )
    p.add_argument(
        "--plot", action="store_true", default=False,
        help="generate bar chart comparison (requires matplotlib)",
    )
    p.add_argument(
        "--plot-output", type=str, default=None,
        help="output path for the plot (default: tp_comparison_<model>.png)",
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

    default_baseline, default_tp = RESULT_FILES[args.model]
    baseline_path = args.baseline or default_baseline
    tp_path = args.tp or default_tp

    try:
        baseline = load_results(baseline_path)
    except FileNotFoundError:
        logger.error("Baseline results not found: %s", baseline_path)
        logger.error("Run the baseline benchmark first (e.g. model_%s.py)", args.model)
        sys.exit(1)

    try:
        tp = load_results(tp_path)
    except FileNotFoundError:
        logger.error("TP results not found: %s", tp_path)
        logger.error("Run the TP benchmark first (e.g. model_%s_tp.py)", args.model)
        sys.exit(1)

    compare(baseline, tp, args.model)

    if args.plot:
        output_path = args.plot_output or f"tp_comparison_{args.model}.png"
        plot_comparison(baseline, tp, args.model, output_path)


if __name__ == "__main__":
    main()
