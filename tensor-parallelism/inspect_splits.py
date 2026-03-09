"""Inspect how weights are split across GPUs in TP mode.

Shows the shape of each parameter on each GPU vs the full (un-split) shape,
and labels whether it is column-parallel, row-parallel, or replicated.
With GQA, K/V projections are smaller than Q projections.

Run:
    torchrun --nproc_per_node=N inspect_splits.py
    torchrun --nproc_per_node=N inspect_splits.py --large
"""

import argparse

import torch
import torch.distributed as dist

from model import LARGE_CONFIG, SMALL_CONFIG
from tp import TPGPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()

    config = LARGE_CONFIG if args.large else SMALL_CONFIG
    tag = "large" if args.large else "small"

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(f"cuda:{rank}")

    model = TPGPT(config).to(f"cuda:{rank}")

    if rank == 0:
        sep = "=" * 85
        gqa = ""
        if config.n_kv_heads < config.n_heads:
            gqa = f" (GQA: {config.n_kv_heads} KV heads)"
        print(f"\n{sep}")
        print(f"  WEIGHT SHAPES - Block 0 - TP={ws} - {tag}{gqa}")
        print(sep)
        print(f"  {'Layer':<42} {'This GPU':<18} {'Full':<18} Note")
        print("  " + "-" * 81)

        for name, p in model.blocks[0].named_parameters():
            full = list(p.shape)
            note = ""
            split = False

            if "W_q" in name and "weight" in name:
                full[0] *= ws
                note = f"<- col-parallel (Q: {config.n_heads} heads)"
                split = True
            elif any(k in name for k in ["W_k", "W_v"]) and "weight" in name:
                full[0] *= ws
                kv_note = "K" if "W_k" in name else "V"
                note = f"<- col-parallel ({kv_note}: {config.n_kv_heads} heads)"
                split = True
            elif "W_o" in name and "weight" in name:
                full[1] *= ws
                note = "<- row-parallel"
                split = True
            elif "W1" in name and "weight" in name:
                full[0] *= ws
                note = "<- col-parallel (d_ff)"
                split = True
            elif "W1" in name and "bias" in name:
                full[0] *= ws
                split = True
            elif "W2" in name and "weight" in name:
                full[1] *= ws
                note = "<- row-parallel"
                split = True
            elif "ln" in name:
                note = "<- replicated"

            tag_str = "[SPLIT]" if split else "[FULL] "
            print(
                f"  {tag_str} {name:<35} "
                f"{str(list(p.shape)):<18} "
                f"{str(full):<18} {note}"
            )

        sharded = sum(
            p.numel()
            for n, p in model.blocks[0].named_parameters()
            if any(k in n for k in ["W_q", "W_k", "W_v", "W_o", "W1", "W2"])
        )
        repl = sum(
            p.numel()
            for n, p in model.blocks[0].named_parameters()
            if "ln" in n
        )
        total_gpu = sharded + repl
        total_full = sharded * ws + repl

        print(sep)
        print(
            f"  On this GPU: {total_gpu:,} | Full block: {total_full:,} | "
            f"Ratio: {total_gpu / total_full * 100:.1f}%"
        )
        print(
            f"  Sharded: {sharded:,} ({sharded / total_gpu * 100:.1f}%) | "
            f"Replicated (LN): {repl:,} ({repl / total_gpu * 100:.1f}%)"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
