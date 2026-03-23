"""TP + DP serving simulation with Qwen2.5-72B.

Demonstrates: 3 replicas x TP=2 on 6 GPUs gives ~3x throughput vs a single
TP=2 instance, while keeping per-request latency the same.

Uses a 2D device mesh: (data_parallel, tensor_parallel).

Run:
    torchrun --nproc_per_node=6 tp_dp_serving.py

Requires: 6 GPUs, transformers.
Outputs: results_tp2_dp3.json
"""

import json
import os
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
MAX_NEW_TOKENS = 64
TP_SIZE = 2
N_PROMPTS_PER_REPLICA = 8

PROMPTS = [
    "Explain the concept of tensor parallelism in distributed computing.",
    "What are the main differences between data parallelism and model parallelism?",
    "Describe how NVLink improves GPU-to-GPU communication bandwidth.",
    "What is the KV cache in transformer inference and why does it matter?",
    "How does grouped-query attention reduce memory usage in large models?",
    "Explain the SwiGLU activation function used in modern LLMs.",
    "What is the difference between prefill and decode phases in LLM serving?",
    "Describe how all-reduce works in a ring topology.",
]


def main():
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{global_rank}")
    torch.cuda.set_device(device)

    n_replicas = world_size // TP_SIZE
    assert world_size == n_replicas * TP_SIZE, (
        f"world_size ({world_size}) must be divisible by TP_SIZE ({TP_SIZE})"
    )

    mesh = init_device_mesh(
        "cuda", (n_replicas, TP_SIZE), mesh_dim_names=("dp", "tp")
    )
    tp_mesh = mesh["tp"]
    dp_mesh = mesh["dp"]
    dp_rank = dp_mesh.get_local_rank()
    tp_rank = tp_mesh.get_local_rank()
    replica_id = dp_rank

    if global_rank == 0:
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  TP x DP Serving: {MODEL_ID}")
        print(f"  {world_size} GPUs = {n_replicas} replicas x TP={TP_SIZE}")
        print(sep)

    # --- Load + parallelize ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    for layer in model.model.layers:
        parallelize_module(
            layer.self_attn,
            tp_mesh,
            {
                "q_proj": ColwiseParallel(),
                "k_proj": ColwiseParallel(),
                "v_proj": ColwiseParallel(),
                "o_proj": RowwiseParallel(),
            },
        )
        parallelize_module(
            layer.mlp,
            tp_mesh,
            {
                "gate_proj": ColwiseParallel(),
                "up_proj": ColwiseParallel(),
                "down_proj": RowwiseParallel(),
            },
        )
    model = model.to(device)

    # Warmup
    warmup_input = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**warmup_input, max_new_tokens=1)
    torch.cuda.synchronize()
    dist.barrier()

    # --- Each replica processes prompts sequentially ---
    torch.cuda.synchronize()
    dist.barrier()
    t_start = time.perf_counter()

    total_tokens = 0
    for i in range(N_PROMPTS_PER_REPLICA):
        prompt = PROMPTS[i % len(PROMPTS)]
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        n_new = output_ids.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += n_new

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t_start
    replica_tps = total_tokens / elapsed

    if tp_rank == 0:
        print(
            f"  Replica {replica_id} "
            f"(GPUs {global_rank}-{global_rank + TP_SIZE - 1}): "
            f"{total_tokens} tokens in {elapsed:.2f}s = {replica_tps:.1f} tok/s"
        )

    tps_tensor = torch.tensor(
        [replica_tps if tp_rank == 0 else 0.0], device=device
    )
    dist.all_reduce(tps_tensor, op=dist.ReduceOp.SUM)
    aggregate_tps = tps_tensor.item()

    if global_rank == 0:
        sep = "=" * 65
        print(f"\n{sep}")
        print("  AGGREGATE THROUGHPUT")
        print(sep)
        print(f"  Config:             {n_replicas} replicas x TP={TP_SIZE}")
        print(f"  Total prompts:      {N_PROMPTS_PER_REPLICA * n_replicas}")
        print(f"  Wall time:          {elapsed:.2f}s")
        print(f"  Aggregate tok/s:    {aggregate_tps:.1f}")
        print(f"  Per-replica tok/s:  {aggregate_tps / n_replicas:.1f}")
        print(sep)

        results = {
            "mode": f"tp{TP_SIZE}_dp{n_replicas}",
            "num_gpus": world_size,
            "tp_size": TP_SIZE,
            "n_replicas": n_replicas,
            "total_prompts": N_PROMPTS_PER_REPLICA * n_replicas,
            "wall_time_s": round(elapsed, 2),
            "aggregate_tokens_per_sec": round(aggregate_tps, 1),
            "per_replica_tokens_per_sec": round(aggregate_tps / n_replicas, 1),
        }
        with open(f"results_tp{TP_SIZE}_dp{n_replicas}.json", "w") as fout:
            json.dump(results, fout, indent=2)
        print(f"  Saved to results_tp{TP_SIZE}_dp{n_replicas}.json")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
