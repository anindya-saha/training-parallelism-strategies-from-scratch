"""Training script using from-scratch Tensor Parallelism.

Supports both small and large model configs, optional sequence parallelism,
and periodic checkpointing.  Designed to be launched via torchrun or
HyperPod PyTorchJob on Kubernetes.

Run locally:
    torchrun --nproc_per_node=2 train_tp.py
    torchrun --nproc_per_node=4 train_tp.py --large --sp --steps 500

Run on K8s (see k8s/hpto_tp.j2.yaml):
    hyperpodrun --nproc_per_node=8 --nnodes=1 train_tp.py --large --sp
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from model import (
    BATCH_SIZE,
    LARGE_CONFIG,
    SEQ_LEN,
    SMALL_CONFIG,
    count_parameters,
    get_gpu_memory_mb,
    get_gpu_peak_memory_mb,
)
from tp import TPGPT, tp_cross_entropy


def parse_args():
    p = argparse.ArgumentParser(description="TP training")
    p.add_argument("--large", action="store_true",
                   help="Use LARGE_CONFIG (d=2048, GQA)")
    p.add_argument("--sp", action="store_true",
                   help="Enable sequence parallelism")
    p.add_argument("--steps", type=int, default=200,
                   help="Total training steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-interval", type=int, default=10,
                   help="Log every N steps")
    p.add_argument("--ckpt-interval", type=int, default=50,
                   help="Checkpoint every N steps")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints",
                   help="Checkpoint directory")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_checkpoint(model, optimizer, step, args, rank):
    if rank != 0:
        return
    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, f"step_{step:06d}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  [ckpt] Saved {path}", flush=True)


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)

    config = LARGE_CONFIG if args.large else SMALL_CONFIG

    if rank == 0:
        sep = "=" * 65
        tag = "LARGE (GQA)" if args.large else "SMALL (MHA)"
        sp_tag = " + SP" if args.sp else ""
        print(f"\n{sep}")
        print(f"  TP Training: {tag}{sp_tag}")
        print(f"  {ws} GPUs, {args.steps} steps, lr={args.lr}")
        print(f"  d_model={config.d_model}, n_heads={config.n_heads}, "
              f"n_kv_heads={config.n_kv_heads}, "
              f"n_layers={config.n_layers}, d_ff={config.d_ff}")
        print(sep, flush=True)

    model = TPGPT(config, sequence_parallel=args.sp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)

    n_params = count_parameters(model)
    if rank == 0:
        print(f"  Params/GPU: {n_params:,}")
        print(f"  Model memory: {get_gpu_memory_mb(device):.1f} MB", flush=True)

    # Training loop with synthetic data
    torch.cuda.reset_peak_memory_stats(device)
    total_tokens = 0
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        input_ids = torch.randint(
            0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device
        )
        labels = torch.randint(
            0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device
        )

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = tp_cross_entropy(logits, labels, config.vocab_size)
        loss.backward()
        optimizer.step()

        total_tokens += BATCH_SIZE * SEQ_LEN

        if rank == 0 and step % args.log_interval == 0:
            elapsed = time.perf_counter() - t_start
            tps = total_tokens / elapsed
            peak = get_gpu_peak_memory_mb(device)
            print(
                f"  step {step:>5}/{args.steps} | "
                f"loss {loss.item():.4f} | "
                f"{tps:,.0f} tok/s | "
                f"peak {peak:.0f} MB",
                flush=True,
            )

        if step % args.ckpt_interval == 0:
            save_checkpoint(model, optimizer, step, args, rank)
            dist.barrier()

    # Final summary
    elapsed = time.perf_counter() - t_start
    peak = get_gpu_peak_memory_mb(device)

    if rank == 0:
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  TRAINING COMPLETE")
        print(sep)
        print(f"  Steps:          {args.steps}")
        print(f"  Wall time:      {elapsed:.1f}s")
        print(f"  Tokens/sec:     {total_tokens / elapsed:,.0f}")
        print(f"  Peak mem/GPU:   {peak:.0f} MB")
        print(f"  Final loss:     {loss.item():.4f}")
        print(sep)

        results = {
            "mode": f"tp_{ws}{'_sp' if args.sp else ''}",
            "config": "large" if args.large else "small",
            "steps": args.steps,
            "wall_time_s": round(elapsed, 1),
            "tokens_per_sec": round(total_tokens / elapsed, 1),
            "peak_mem_mb": round(peak, 1),
            "final_loss": round(loss.item(), 4),
        }
        with open("results_train_tp.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results_train_tp.json")

    save_checkpoint(model, optimizer, args.steps, args, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
