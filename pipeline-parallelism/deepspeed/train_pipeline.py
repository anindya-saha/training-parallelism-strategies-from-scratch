#!/usr/bin/env python3
"""
DeepSpeed Pipeline Parallelism Training
Uses PipelineModule to split the model across 4 GPUs.
"""

import os, time, json, argparse
import torch
import torch.distributed as dist
import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from gpt2_model import (
    get_layers, lm_loss_fn, count_parameters,
    VOCAB_SIZE, MAX_SEQ_LEN, NUM_LAYERS
)


class SyntheticLMData(torch.utils.data.Dataset):
    """Synthetic token data for benchmarking."""

    def __init__(self, seq_len, num_samples=10000):
        self.data = torch.randint(0, VOCAB_SIZE, (num_samples, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return tokens[:-1].long(), tokens[1:].long()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--tag", type=str, default="pp4")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Build pipeline model
    layers = get_layers()
    total_params = sum(sum(p.numel() for p in layer.parameters()) for layer in layers)

    model = PipelineModule(
        layers=layers,
        loss_fn=lm_loss_fn,
        num_stages=world_size,
        partition_method="parameters",    # Balance by parameter count
        activation_checkpoint_interval=0, # No activation checkpointing
    )

    # Data loader
    dataset = SyntheticLMData(seq_len=args.seq_len)

    engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters()],
        training_data=dataset,
    )

    train_iter = iter(RepeatingLoader(train_loader))

    if rank == 0:
        stage_params = count_parameters(model)
        print(f"\n{'='*60}")
        print(f"  PIPELINE PARALLELISM: {world_size} stages [{args.tag}]")
        print(f"  Total model: {total_params/1e9:.2f}B params")
        print(f"  This stage (rank {rank}): {stage_params/1e6:.0f}M params")
        print(f"  Micro-batch size: {engine.train_micro_batch_size_per_gpu()}")
        print(f"  Gradient accum steps: {engine.gradient_accumulation_steps()}")
        print(f"  Global batch size: {engine.train_batch_size()}")
        print(f"{'='*60}\n")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(local_rank)

    # Training loop
    step_times = []
    losses = []

    for step in range(args.warmup_steps + args.num_steps):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()

        loss = engine.train_batch(data_iter=train_iter)

        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter()

        if step >= args.warmup_steps:
            step_times.append(t1 - t0)
            if engine.is_last_stage():
                losses.append(loss.item())

        if rank == 0 and step % 5 == 0:
            loss_val = loss.item() if engine.is_last_stage() else 0.0
            print(f"  Step {step:3d} | Loss: {loss_val:.4f} | "
                  f"Time: {(t1-t0)*1000:.1f}ms")

    # Gather results
    peak_mem = torch.cuda.max_memory_allocated(local_rank) / 1e9
    all_mems = [None] * world_size
    dist.all_gather_object(all_mems, peak_mem)

    if rank == 0:
        avg_time = sum(step_times) / len(step_times)
        global_batch = engine.train_batch_size()
        throughput = global_batch * args.seq_len / avg_time

        results = {
            "experiment": f"pipeline_{args.tag}",
            "num_stages": world_size,
            "micro_batch_size": engine.train_micro_batch_size_per_gpu(),
            "gradient_accumulation_steps": engine.gradient_accumulation_steps(),
            "global_batch_size": global_batch,
            "avg_step_time_ms": avg_time * 1000,
            "throughput_tokens_per_sec": throughput,
            "peak_gpu_memory_gb": all_mems,
            "max_gpu_memory_gb": max(all_mems),
            "step_times_ms": [t * 1000 for t in step_times],
        }

        print(f"\n{'='*60}")
        print(f"  RESULTS: Pipeline Parallelism ({args.tag})")
        print(f"  Avg step time:   {avg_time*1000:.1f} ms")
        print(f"  Throughput:      {throughput:,.0f} tokens/sec")
        print(f"  GPU memory per stage:")
        for r, m in enumerate(all_mems):
            print(f"    GPU {r}: {m:.2f} GB")
        print(f"{'='*60}\n")

        fname = f"results_{args.tag}.json"
        with open(fname, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {fname}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
