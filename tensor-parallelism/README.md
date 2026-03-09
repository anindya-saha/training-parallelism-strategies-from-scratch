# Tensor Parallelism

From-scratch implementation of Tensor Parallelism (TP), plus production
inference scripts using PyTorch native TP on a 72B model.

Supports GQA (Grouped Query Attention), Sequence Parallelism (SP), and
combined TP + FSDP with a 2D device mesh.

## Scripts

### From-Scratch TP (training)

| Script | What it does | How to run |
|--------|-------------|------------|
| `model.py` | Standard GPT + shared constants + ModelConfig | imported by others |
| `tp.py` | TP primitives: ColumnParallelLinear, RowParallelLinear, TPGPT | imported by others |
| `benchmark_baseline.py` | Single-GPU baseline (memory + throughput) | `python benchmark_baseline.py [--large]` |
| `benchmark_tp.py` | Multi-GPU TP benchmark | `torchrun --nproc_per_node=N benchmark_tp.py [--large] [--sp]` |
| `inspect_splits.py` | Shows weight shapes per GPU (split vs replicated) | `torchrun --nproc_per_node=N inspect_splits.py [--large]` |
| `measure_allreduce.py` | All-reduce latency microbenchmark | `torchrun --nproc_per_node=N measure_allreduce.py` |
| `compare_results.py` | Side-by-side comparison of baseline vs TP | `python compare_results.py` |
| `train_tp.py` | Full training loop with TP (K8s-deployable) | `torchrun --nproc_per_node=N train_tp.py [--large] [--sp]` |
| `train_tp_fsdp.py` | TP + FSDP combined (2D device mesh) | `torchrun --nproc_per_node=4 train_tp_fsdp.py --tp-size 2` |

### Production TP (inference with Qwen 72B)

| Script | What it does | How to run |
|--------|-------------|------------|
| `tp_inference_72b.py` | TP inference on Qwen2.5-72B | `torchrun --nproc_per_node=4 tp_inference_72b.py` |
| `tp_dp_serving.py` | TP + DP: 3 replicas x TP=2 on 6 GPUs | `torchrun --nproc_per_node=6 tp_dp_serving.py` |

## Model Configurations

Two preset configs in `model.py`:

| Config | d_model | n_heads | n_kv_heads | d_ff | n_layers | Attention | Params |
|--------|---------|---------|------------|------|----------|-----------|--------|
| `SMALL_CONFIG` | 512 | 8 | 8 | 2048 | 6 | MHA | ~29M |
| `LARGE_CONFIG` | 2048 | 16 | 4 | 8192 | 12 | GQA | ~350M |

LARGE_CONFIG uses GQA (4 KV heads serving 16 Q heads), matching the pattern
used by Llama 3, Qwen 2.5, and other modern LLMs.

**TP degree constraints:** must divide both `n_heads` and `n_kv_heads`.
- SMALL: TP = 2, 4, or 8
- LARGE: TP = 1, 2, or 4 (n_kv_heads=4 limits max TP)

## Quick Start

```bash
# 1. Baseline (single GPU)
python benchmark_baseline.py

# 2. TP with 2 GPUs
torchrun --nproc_per_node=2 benchmark_tp.py

# 3. Compare
python compare_results.py

# 4. TP + Sequence Parallelism
torchrun --nproc_per_node=2 benchmark_tp.py --sp

# 5. Large model with GQA
python benchmark_baseline.py --large
torchrun --nproc_per_node=4 benchmark_tp.py --large

# 6. Inspect weight splits (GQA shows K/V smaller than Q)
torchrun --nproc_per_node=2 inspect_splits.py --large

# 7. Full training with TP
torchrun --nproc_per_node=2 train_tp.py --steps 100

# 8. TP + FSDP on 4 GPUs (TP=2 x FSDP=2)
torchrun --nproc_per_node=4 train_tp_fsdp.py --tp-size 2

# 9. Measure communication overhead
torchrun --nproc_per_node=2 measure_allreduce.py
```

## Architecture

### TP Communication Pattern

```
TP splits individual weight matrices across GPUs:

  Column-Parallel (expanding layers):     Row-Parallel (contracting layers):
    W_q, W_k, W_v  -- split heads          W_o         -- split input dim
    W1 (FFN up)     -- split d_ff           W2 (FFN down) -- split input dim
    No communication in forward             ALL-REDUCE in forward

  Communication per transformer block: 2 all-reduces
    1. After attention (W_o row-parallel)
    2. After FFN (W2 row-parallel)

  Replicated (NOT split): LayerNorm, embeddings
```

### Sequence Parallelism (SP)

```
Without SP:                         With SP:
  LN input: (B, T, d) replicated     LN input: (B, T/N, d) seq-split
  -> all-reduce after W_o             -> all-gather before Q/K/V
  -> all-reduce after W2              -> reduce-scatter after W_o/W2

  Same total comm volume, but LN/residual activations use 1/N memory.
```

### GQA (Grouped Query Attention)

```
MHA: Q(8 heads), K(8 heads), V(8 heads)   -- all heads independent
GQA: Q(16 heads), K(4 heads), V(4 heads)  -- 4 Q heads share each KV head

With TP=4 on LARGE_CONFIG (16 Q, 4 KV heads):
  GPU 0: Q heads 0-3,  KV head 0  (repeat KV 4x before attention)
  GPU 1: Q heads 4-7,  KV head 1
  GPU 2: Q heads 8-11, KV head 2
  GPU 3: Q heads 12-15, KV head 3
```

### 2D Parallelism (TP + FSDP)

```
2 nodes x 4 GPUs/node = 8 GPUs total:

  mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

  Node 0: [GPU 0 -- GPU 1 -- GPU 2 -- GPU 3]   TP group (NVLink)
  Node 1: [GPU 4 -- GPU 5 -- GPU 6 -- GPU 7]   TP group (NVLink)
            |         |         |         |
            +-------- FSDP (cross-node) ---+

  TP handles compute splitting within a node.
  FSDP handles memory optimization across nodes.
```

## Kubernetes Deployment

See `k8s/hpto_tp.j2.yaml` for a HyperPod PyTorchJob template. Render with:

```bash
jinja2 k8s/hpto_tp.j2.yaml \
  -D JOB_NAME=tp-train-large \
  -D NAMESPACE=mlp \
  -D IMAGE_URI=your-registry/tp-training:latest \
  -D INSTANCE_TYPE=ml.p4d.24xlarge \
  -D SERVICE_ACCOUNT=mlp-sa \
  -D USE_LARGE=true \
  -D USE_SP=true \
  -D STEPS=1000 | kubectl apply -f -
```

## Key Results (from Vizuara tutorial on 2x H200)

```
                         No TP (1 GPU)     TP (2 GPUs)
  Parameters/GPU            29,405,184      17,401,856  (40.8% fewer)
  Model memory (MB)              113.1            66.8  (40.9% less)
  Peak memory (MB)              1204.4           837.4  (30.5% less)
  Full step (ms)                 11.84           19.19
  Efficiency                       ---           30.8%
```

TP reduces memory but adds communication overhead. Efficiency improves with
larger models where compute dominates over all-reduce latency.

## References

- [Megatron-LM paper](https://arxiv.org/abs/1909.08053) -- original TP formulation
- [Megatron-LM v3](https://arxiv.org/abs/2205.05198) -- sequence parallelism
- [GQA paper](https://arxiv.org/abs/2305.13245) -- grouped query attention
- Vizuara GPU Engineering Course -- source material for these implementations
- `tutorials/vizuara/06-all-about-tensor-parallelism/` -- original notebooks
