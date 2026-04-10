# Context Parallelism Tutorial

## Quick Start

Run these 3 commands in order:

```bash
# Step 1: Verify NCCL works
torchrun --nproc_per_node=2 step1_test_nccl.py

# Step 2: Run CP comparison
torchrun --nproc_per_node=2 step2_cp_comparison.py

# Step 3: Generate plots
pip install matplotlib -q && python step3_plot.py
```

## What This Demonstrates

**The Problem:** Attention memory is O(S²) — explodes with sequence length!

| Sequence | Attention Matrix | Memory |
|----------|------------------|--------|
| 1K | 1K × 1K | 64 MB |
| 4K | 4K × 4K | 1 GB |
| 16K | 16K × 16K | 16 GB |
| 64K | 64K × 64K | 256 GB ← Impossible! |

**The Solution:** Context Parallelism splits queries across GPUs.

```
Without CP:  Each GPU computes (S × S) attention
With CP=2:   Each GPU computes (S/2 × S) attention → 2x smaller!
With CP=4:   Each GPU computes (S/4 × S) attention → 4x smaller!
```

## Expected Output

```
=================================================================
  CONTEXT PARALLELISM COMPARISON
=================================================================
  GPUs: 2

--- Sequence Length: 1024 ---
  No CP:   attn=(1024×1024)     mem=  XXX.X MB  time=  X.XX ms
  CP=2:    attn=(512×1024)      mem=  XXX.X MB  time=  X.XX ms
  Reduction: ~50%

--- Sequence Length: 2048 ---
  No CP:   attn=(2048×2048)     mem=  XXX.X MB  time=  X.XX ms
  CP=2:    attn=(1024×2048)     mem=  XXX.X MB  time=  X.XX ms
  Reduction: ~50%
```

## Files

| File | Description |
|------|-------------|
| `step1_test_nccl.py` | Verify distributed works |
| `step2_cp_comparison.py` | Main CP benchmark |
| `step3_plot.py` | Generate comparison plots |
| `cp_results.json` | Results (generated) |
| `cp_comparison.png` | Plots (generated) |
