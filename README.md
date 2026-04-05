# Parallelism Strategies for Training Large Language Models

Understanding scaling LLM training - with concrete code, hand-drawn diagrams, and reproducible benchmarks.

> Inspired by the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultra-scale-playbook)
> by Hugging Face. Rewritten based on the HF blog with original implementations,
> diagrams, and numerical walkthroughs.

---

## Table of Contents

1. [High Level Overview](#high-level-overview)
2. [First Steps: Training on One GPU](#first-steps-training-on-one-gpu)
3. [Data Parallelism](#data-parallelism)
4. [Tensor Parallelism](#tensor-parallelism)
5. [Sequence Parallelism](#sequence-parallelism)
6. [Context Parallelism](#context-parallelism)
7. [Pipeline Parallelism](#pipeline-parallelism)
8. [Expert Parallelism](#expert-parallelism)
9. [5D Parallelism in a Nutshell](#5d-parallelism-in-a-nutshell)
10. [Finding the Best Training Configuration](#finding-the-best-training-configuration)
11. [Diving in the GPUs - Fusing, Threading, Mixing](#diving-in-the-gpus----fusing-threading-mixing)
12. [Appendix](#appendix)

---


## High Level Overview

TODO


---


## Running Tests and Benchmarks

See [developer.md](developer.md) for setup, CLI flags, benchmark commands, and Kubernetes deployment.


---


## First Steps: Training on One GPU

TODO

### Memory Usage in Transformers

TODO

### Activation Recomputation

TODO

### Gradient Accumulation

TODO


---


## Data Parallelism

TODO

### Revisit Global Batch Size

TODO

### Our Journey Up to Now

TODO

### ZeRO (Zero Redundancy Optimizer)

TODO


---


## Tensor Parallelism

See [tensor-parallelism.md](tensor-parallelism.md) for the full writeup: Column-Parallel and Row-Parallel Linear derivations with concrete matrix examples, autograd primitives, the conjugate cancellation, Transformer block TP (attention + MLP), MHA/GQA/MQA comparison, and communication summary.

## Sequence Parallelism

See [sequence-parallelism.md](sequence-parallelism.md) for the full write-up on Sequence Parallelism.
---


## Context Parallelism

TODO

### Discovering Ring Attention

TODO

### Zig-Zag Ring Attention

TODO


---


## Pipeline Parallelism

TODO

### Splitting Layers on Various Nodes - All Forward, All Backward

TODO

### One-Forward-One-Backward and LLama 3.1 Schemes

TODO

### Interleaving Stages

TODO

### Zero Bubble and DualPipe

TODO


---


## Expert Parallelism

TODO


---


## 5D Parallelism in a Nutshell

TODO


---


## Finding the Best Training Configuration

TODO

### Step 1: Fitting a Training Step in Memory

TODO

### Step 2: Achieving Target Global Batch Size

TODO

### Step 3: Optimizing Training Throughput

TODO

### Benchmarking Thousands of Configurations

TODO

### Lessons Learned on Benchmarking

TODO


---


## Diving in the GPUs - Fusing, Threading, Mixing

TODO

### A Primer on GPU

TODO

### How to Improve Performance with Kernels

TODO

### Fused Kernels

TODO

### Flash Attention 1-3

TODO

### Mixed Precision Training

TODO


---


## Appendix

### A0: Parallel Programming Crash Course

TODO

### A1: Distributed Training Profiling

TODO

### A2: Typical Scales in LLM Training

TODO

### A3: Math for Compute/Communication Overlap

TODO


---


## References

TODO


---

> Adapted from the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultra-scale-playbook)
> (Apache 2.0) by Hugging Face. All code, diagrams, and numerical examples are
> original work.
