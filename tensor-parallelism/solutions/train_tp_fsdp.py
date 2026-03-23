"""Combined TP + FSDP training using a 2D device mesh.

TP (Tensor Parallelism) splits individual weight matrices across GPUs within
a node, communicating via fast NVLink.  FSDP (Fully Sharded Data Parallel)
shards full parameters + optimizer states across nodes, communicating via
slower inter-node fabric.

2D Device Mesh layout (example: 2 nodes x 4 GPUs/node = 8 GPUs total):

    Node 0: [GPU 0, GPU 1, GPU 2, GPU 3]  <-- TP group (NVLink)
    Node 1: [GPU 4, GPU 5, GPU 6, GPU 7]  <-- TP group (NVLink)
              |       |       |       |
              +-- FSDP group (cross-node) --+

    mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

Supports optional sequence parallelism within the TP dimension.

Run locally (simulated 2D on a single node):
    torchrun --nproc_per_node=4 train_tp_fsdp.py --tp-size 2
    # This creates: FSDP=2 x TP=2 on 4 GPUs

Multi-node:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$RANK \\
        --master_addr=$MASTER --master_port=29500 \\
        train_tp_fsdp.py --tp-size 8
    # This creates: FSDP=2 (across nodes) x TP=8 (within node)
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

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
    p = argparse.ArgumentParser(description="TP + FSDP training")
    p.add_argument("--large", action="store_true")
    p.add_argument("--sp", action="store_true",
                   help="Enable sequence parallelism within TP groups")
    p.add_argument("--tp-size", type=int, default=2,
                   help="TP degree (GPUs per TP group, typically = GPUs/node)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--ckpt-interval", type=int, default=50)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_tp_fsdp")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_checkpoint(model, optimizer, step, args, global_rank):
    if global_rank != 0:
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
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{global_rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + global_rank)

    tp_size = args.tp_size
    dp_size = world_size // tp_size
    assert world_size == dp_size * tp_size, (
        f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
    )

    config = LARGE_CONFIG if args.large else SMALL_CONFIG

    # ----------------------------------------------------------------
    # 2D device mesh: (dp, tp)
    # ----------------------------------------------------------------
    mesh_2d = init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    tp_mesh = mesh_2d["tp"]
    dp_mesh = mesh_2d["dp"]
    tp_pg = tp_mesh.get_group()
    dp_pg = dp_mesh.get_group()

    tp_rank = dist.get_rank(tp_pg)
    dp_rank = dist.get_rank(dp_pg)

    if global_rank == 0:
        sep = "=" * 65
        tag = "LARGE (GQA)" if args.large else "SMALL (MHA)"
        sp_tag = " + SP" if args.sp else ""
        print(f"\n{sep}")
        print(f"  TP + FSDP Training: {tag}{sp_tag}")
        print(f"  {world_size} GPUs = FSDP({dp_size}) x TP({tp_size})")
        print(f"  d_model={config.d_model}, n_heads={config.n_heads}, "
              f"n_kv_heads={config.n_kv_heads}, "
              f"n_layers={config.n_layers}, d_ff={config.d_ff}")
        print(sep, flush=True)

    # ----------------------------------------------------------------
    # Build model with TP using the TP process group
    # ----------------------------------------------------------------
    model = TPGPT(
        config,
        sequence_parallel=args.sp,
        process_group=tp_pg,
    ).to(device)

    tp_params = count_parameters(model)
    if global_rank == 0:
        print(f"  Params/GPU (after TP): {tp_params:,}", flush=True)

    # ----------------------------------------------------------------
    # Wrap with FSDP using the DP process group
    # ----------------------------------------------------------------
    model = FSDP(
        model,
        process_group=dp_pg,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)

    fsdp_params = count_parameters(model)
    if global_rank == 0:
        print(f"  Params/GPU (after FSDP): {fsdp_params:,}")
        print(f"  Model memory: {get_gpu_memory_mb(device):.1f} MB", flush=True)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
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
        loss = tp_cross_entropy(logits, labels, config.vocab_size,
                                process_group=tp_pg)
        loss.backward()
        optimizer.step()

        total_tokens += BATCH_SIZE * SEQ_LEN

        if global_rank == 0 and step % args.log_interval == 0:
            elapsed = time.perf_counter() - t_start
            tps = total_tokens / elapsed
            peak = get_gpu_peak_memory_mb(device)
            print(
                f"  step {step:>5}/{args.steps} | "
                f"loss {loss.item():.4f} | "
                f"{tps:,.0f} tok/s | "
                f"peak {peak:.0f} MB | "
                f"dp_rank={dp_rank} tp_rank={tp_rank}",
                flush=True,
            )

        if step % args.ckpt_interval == 0:
            save_checkpoint(model, optimizer, step, args, global_rank)
            dist.barrier()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    peak = get_gpu_peak_memory_mb(device)

    if global_rank == 0:
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  TRAINING COMPLETE (TP={tp_size} x FSDP={dp_size})")
        print(sep)
        print(f"  Steps:          {args.steps}")
        print(f"  Wall time:      {elapsed:.1f}s")
        print(f"  Tokens/sec:     {total_tokens / elapsed:,.0f}")
        print(f"  Peak mem/GPU:   {peak:.0f} MB")
        print(f"  Final loss:     {loss.item():.4f}")
        print(sep)

        results = {
            "mode": f"tp{tp_size}_fsdp{dp_size}{'_sp' if args.sp else ''}",
            "config": "large" if args.large else "small",
            "num_gpus": world_size,
            "tp_size": tp_size,
            "dp_size": dp_size,
            "steps": args.steps,
            "wall_time_s": round(elapsed, 1),
            "tokens_per_sec": round(total_tokens / elapsed, 1),
            "peak_mem_mb": round(peak, 1),
            "final_loss": round(loss.item(), 4),
        }
        with open("results_train_tp_fsdp.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results_train_tp_fsdp.json")

    save_checkpoint(model, optimizer, args.steps, args, global_rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
