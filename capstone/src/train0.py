"""
Training script for Vizz SLIM.
Refactored from Karpathy's build-nanogpt/train_gpt2.py.

Usage:
  Single GPU:      python train.py
  DDP (8 GPUs):    torchrun --standalone --nproc_per_node=8 train.py
  With parallelism: torchrun --standalone --nproc_per_node=8 train.py --tp_size 2 --pp_size 2
"""
import os
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from config import ModelConfig, ParallelConfig, TrainConfig
from model import GPT, apply_tensor_parallelism, apply_context_parallelism, apply_moe
from data import DataLoaderLite


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vizz SLIM")
    # Parallelism config
    parser.add_argument('--tp_size', type=int, default=1, help='Tensor Parallelism degree')
    parser.add_argument('--pp_size', type=int, default=1, help='Pipeline Parallelism degree')
    parser.add_argument('--cp_size', type=int, default=1, help='Context Parallelism degree')
    parser.add_argument('--ep_size', type=int, default=1, help='Expert Parallelism degree')
    # Model config
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--num_experts', type=int, default=0, help='0=dense, >0=MoE')
    parser.add_argument('--moe_top_k', type=int, default=2)
    # Training config
    parser.add_argument('--total_batch_size', type=int, default=524288)
    parser.add_argument('--micro_batch_size', type=int, default=64)
    parser.add_argument('--max_lr', type=float, default=6e-4)
    parser.add_argument('--max_steps', type=int, default=19073)
    parser.add_argument('--warmup_steps', type=int, default=715)
    parser.add_argument('--data_root', type=str, default='edu_fineweb10B')
    parser.add_argument('--eval_interval', type=int, default=250)
    return parser.parse_args()


def get_lr(it, warmup_steps, max_steps, max_lr, min_lr):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    args = parse_args()

    # ---- Distributed setup ----
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA required for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        if master_process:
            print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # ---- Parallelism config ----
    par_config = ParallelConfig(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        cp_size=args.cp_size,
        ep_size=args.ep_size,
    )

    # For pure DP mode (all parallelism sizes = 1), use simple DDP
    use_simple_dp = (par_config.tp_size == 1 and par_config.pp_size == 1 and
                     par_config.cp_size == 1 and par_config.ep_size == 1)

    if use_simple_dp:
        # Simple DDP mode - same as original nanogpt
        dp_world_size = ddp_world_size
        dp_rank = ddp_rank
        groups = {'tp': None, 'pp': None, 'cp': None, 'ep': None, 'dp': None}
    else:
        # 5D parallelism mode
        par_config.validate(ddp_world_size)
        from utils import create_process_groups
        groups = create_process_groups(par_config)
        dp_world_size = par_config.dp_size
        from utils import get_data_parallel_rank
        dp_rank = get_data_parallel_rank(groups)

    # ---- Model config ----
    model_config = ModelConfig(
        block_size=args.block_size,
        vocab_size=50304,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
    )

    # ---- Training config ----
    B = args.micro_batch_size
    T = model_config.block_size
    total_batch_size = args.total_batch_size
    assert total_batch_size % (B * T * dp_world_size) == 0, \
        f"total_batch_size ({total_batch_size}) must be divisible by B*T*dp_size ({B}*{T}*{dp_world_size})"
    grad_accum_steps = total_batch_size // (B * T * dp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        print(f"parallelism: TP={par_config.tp_size} PP={par_config.pp_size} "
              f"CP={par_config.cp_size} EP={par_config.ep_size} DP={dp_world_size}")

    # ---- Data ----
    train_loader = DataLoaderLite(
        B=B, T=T, process_rank=dp_rank, num_processes=dp_world_size,
        split="train", data_root=args.data_root, master_process=master_process
    )
    val_loader = DataLoaderLite(
        B=B, T=T, process_rank=dp_rank, num_processes=dp_world_size,
        split="val", data_root=args.data_root, master_process=master_process
    )

    torch.set_float32_matmul_precision('high')

    # ---- Build model ----
    if par_config.pp_size > 1:
        # Pipeline parallelism: use PipelineStage instead of full GPT
        from parallel.pipeline import create_pipeline_stages
        stage = create_pipeline_stages(model_config, par_config.pp_size, groups['pp'], device)
        stage = stage.to(device)
        raw_model = stage
        use_pipeline = True
    else:
        model = GPT(model_config)
        model.to(device)
        use_pipeline = False

        # Apply parallelism dimensions
        if par_config.tp_size > 1:
            apply_tensor_parallelism(model, groups['tp'])
        if par_config.cp_size > 1:
            apply_context_parallelism(model, groups['cp'])
        if model_config.num_experts > 0:
            apply_moe(model, model_config, groups['ep'] if par_config.ep_size > 1 else None)

        # Wrap with DDP for data parallelism
        if ddp and dp_world_size > 1:
            use_find_unused = model_config.num_experts > 0
            model = DDP(model, device_ids=[ddp_local_rank],
                        process_group=groups['dp'],
                        find_unused_parameters=use_find_unused)
        raw_model = model.module if (ddp and dp_world_size > 1) else model

    # ---- Optimizer ----
    max_lr = args.max_lr
    min_lr = max_lr * 0.1
    warmup_steps = args.warmup_steps
    max_steps = args.max_steps
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=max_lr,
        device_type=device_type, master_process=master_process
    )

    # ---- Logging ----
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    if master_process:
        with open(log_file, "w") as f:
            pass

    # ---- Training loop ----
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # ---- Validation ----
        if step % args.eval_interval == 0 or last_step:
            if not use_pipeline:
                model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = torch.tensor(0.0, device=device)
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    if use_pipeline:
                        # Pipeline eval - simplified
                        pass
                    else:
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(x, y)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # ---- Training step ----
        if not use_pipeline:
            model.train()
        optimizer.zero_grad()
        loss_accum = torch.tensor(0.0, device=device)

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if use_pipeline:
                # Pipeline parallelism training step
                from parallel.pipeline import pipeline_schedule_1f1b
                loss = pipeline_schedule_1f1b(
                    stage, [x], [y], groups['pp'], device
                )
                if loss is not None:
                    loss_accum += loss.detach() / grad_accum_steps
            else:
                # Standard forward/backward with DDP gradient sync control
                if ddp and dp_world_size > 1:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)

        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = B * T * grad_accum_steps * dp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
                  f"norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
