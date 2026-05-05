## From Hand-Written Pipeline Parallelism to torch.distributed.pipelining

*This is Part 2 of a two-part series on Pipeline Parallelism. [Part 1](pipeline-parallelism.md) builds hand-written Naive, GPipe, and 1F1B schedules from scratch with manual `dist.send`/`dist.recv`. Parts [1](tensor-parallelism-blog.md)/[2](tensor-parallelism-dtensor.md) cover Tensor Parallelism. Parts [3](sequence-parallelism-blog.md)/[4](sequence-parallelism-dtensor.md) cover Sequence Parallelism. Parts [5](context-parallelism-blog.md)/[6](context-parallelism-dtensor.md) cover Context Parallelism. Parts [7](expert-parallelism-blog.md)/[8](expert-parallelism-ep-blog.md)/[9](expert-parallelism-ep-dtensor.md) cover Expert Parallelism.*

In [Part 1](pipeline-parallelism.md) we built three pipeline schedules by hand: naive (one batch, maximum bubble), GPipe (all-forwards then all-backwards), and 1F1B (warmup/steady/cooldown). Each script manages its own `dist.send`/`dist.recv` calls, microbatch splitting, activation storage, and gradient exchange. The logic works, but every schedule requires 50-80 lines of careful send/recv ordering that must match between adjacent ranks -- a deadlock if any pair disagrees on the protocol.

This article replaces all of that with `torch.distributed.pipelining`: [train_gpt_pp_pipelining.py](pipeline-parallelism/src/train_gpt_pp_pipelining.py). The model code is **identical** to a single-GPU GPT. Partitioning still happens via the same `get_stage` pattern (meta device, slice layers, materialize). But instead of writing send/recv loops, we wrap the stage in a `PipelineStage` and pick a schedule (`ScheduleGPipe` or `Schedule1F1B`) -- the framework handles microbatch splitting, communication buffer allocation, and gradient propagation automatically.


### What torch.distributed.pipelining Provides

Three components from `torch.distributed.pipelining` make the framework approach possible:

**1. `PipelineStage` -- wraps an `nn.Module` with communication plumbing.** It allocates send/recv buffers based on the stage's input/output shapes, manages intermediate activation buffers, and provides utilities for running forward and backward. The stage module itself has no knowledge of pipeline communication.

```python
from torch.distributed.pipelining import PipelineStage

stage = PipelineStage(
    stage_module,          # nn.Sequential (our local layers)
    stage_index=rank,      # which stage this is (0-indexed)
    num_stages=pp_size,    # total pipeline depth
    device=device,
    input_args=example_input,   # for shape/dtype inference
    output_args=example_output, # output shape/dtype
)
```

**2. `ScheduleGPipe` / `Schedule1F1B` -- schedule implementations.** These classes take a `PipelineStage` and a microbatch count, then execute the full schedule (forward + backward for all microbatches) in a single `.step()` call. Internally they handle microbatch splitting, the send/recv ordering, and gradient accumulation.

**3. `loss_fn` parameter -- loss computation on the last stage.** Instead of manually computing loss and calling backward, the schedule accepts a `loss_fn` that it applies to the last stage's output. This integrates loss computation into the schedule's backward pass.


### The Key Contrast: Model Code Side by Side

In Part 1, the schedule logic is interleaved with communication:

```python
# Part 1: Hand-written GPipe -- schedule is manual send/recv

def pipeline_forward_backward_gpipe(stage, rank, pp_size, micro_inputs, d_model, device):
    # Forward phase: N-stage chain of send/recv per microbatch
    for b in range(n_micro):
        if is_first:
            h = stage(micro_inputs[b])
            dist.send(h.detach().contiguous(), dst=rank + 1)     # manual send
        elif is_last:
            buf = torch.empty(mbs, T, d_model, device=device)
            dist.recv(buf, src=rank - 1)                         # manual recv
            h_req = buf.clone().requires_grad_(True)
            logits = stage(h_req)
            # ... save for backward
        else:
            # recv, forward, send -- intermediate rank logic
            ...

    # Backward phase: reverse order, send grads back
    for b in range(n_micro - 1, -1, -1):
        if is_last:
            loss_b = lm_loss(logits, ids_b)
            loss_b.backward()
            dist.send(h_req.grad.detach().contiguous(), dst=rank - 1)  # manual grad send
        elif is_first:
            grad_h = torch.empty(mbs, T, d_model, device=device)
            dist.recv(grad_h, src=rank + 1)                            # manual grad recv
            saved_outputs[b].backward(grad_h)
        else:
            # recv grad, backward, send grad -- intermediate rank logic
            ...
```

In Part 2, the entire schedule is one line:

```python
# Part 2: torch.distributed.pipelining -- schedule is one call

schedule = ScheduleGPipe(pipeline_stage, n_microbatches=n_micro, loss_fn=lm_loss)

# Training step -- the framework handles everything
if rank == 0:
    schedule.step(input_ids)        # first stage provides input
elif rank == pp_size - 1:
    losses = []
    schedule.step(target=input_ids, losses=losses)  # last stage gets target + loss list
else:
    schedule.step()                 # intermediate stages: no arguments needed
```

No `dist.send`, no `dist.recv`, no activation storage lists, no microbatch splitting logic, no forward/backward ordering. The framework manages it all.


### How PipelineStage Works

`PipelineStage` is responsible for:

1. **Buffer allocation**: At construction time, it uses `input_args` and `output_args` to determine the tensor shapes and dtypes for communication. It pre-allocates send/recv buffers so no allocation happens during training.

2. **Send/recv operations**: During execution, it creates the appropriate `dist.send`/`dist.recv` operations to communicate with peer stages (rank-1 for receiving activations, rank+1 for sending activations, and the reverse for gradients).

3. **Activation management**: It handles storing intermediate activations from forward passes that are needed for backward, implementing the same logic we wrote by hand in `saved_outputs` and `saved_inputs` lists.

The key constraint: **input and output shapes must be static** (known at construction time). This is why we pass `input_args` and `output_args` -- the stage needs to know buffer sizes before training starts. Dynamic shapes would require reallocating communication buffers every step.

```python
# Shape inference for PipelineStage
if is_first:
    # First stage receives input_ids: (mbs, seq_len) int64
    example_input = (torch.randint(0, vocab_size, (mbs, seq_len), device=device),)
else:
    # Other stages receive hidden states: (mbs, seq_len, d_model) float32
    example_input = (torch.randn(mbs, seq_len, d_model, device=device),)

if is_last:
    # Last stage outputs logits: (mbs, seq_len, vocab_size) float32
    output_args = (torch.randn(mbs, seq_len, vocab_size, device=device),)
else:
    # Other stages output hidden states: (mbs, seq_len, d_model) float32
    output_args = (torch.randn(mbs, seq_len, d_model, device=device),)
```


### Schedule Selection: GPipe vs 1F1B

Both schedules are drop-in replacements -- same `PipelineStage`, same `.step()` API:

```python
# GPipe: all-forwards then all-backwards
schedule = ScheduleGPipe(pipeline_stage, n_microbatches=n_micro, loss_fn=lm_loss)

# 1F1B: warmup/steady/cooldown interleaving
schedule = Schedule1F1B(pipeline_stage, n_microbatches=n_micro, loss_fn=lm_loss)
```

The memory vs throughput tradeoff is the same as in Part 1:
- **GPipe**: Peak activations = M (all microbatch outputs stored until backward phase)
- **1F1B**: Peak activations = `pp_size - 1 - rank + 1` on rank r (bounded by pipeline depth)

But in the hand-written version, switching from GPipe to 1F1B requires rewriting the entire schedule function (~80 lines). With the framework, it is a one-line change.


### The Payoff: What Disappears

| Hand-written primitive (Part 1) | Lines | Framework equivalent (Part 2) | Lines |
|---|---|---|---|
| `dist.send` / `dist.recv` for activations | ~15 per schedule | `PipelineStage` handles internally | 0 |
| `dist.send` / `dist.recv` for gradients | ~15 per schedule | `PipelineStage` handles internally | 0 |
| Microbatch splitting logic | ~5 | Framework splits automatically | 0 |
| Activation storage (`saved_outputs`, `saved_inputs`) | ~10 | Framework manages buffers | 0 |
| Forward/backward ordering (schedule logic) | ~30-50 | `ScheduleGPipe` or `Schedule1F1B` | 1 |
| Buffer pre-allocation (`torch.empty`) | ~8 | `input_args`/`output_args` at construction | 3 |
| Loss computation + backward integration | ~5 | `loss_fn` parameter | 0 |
| **Total schedule-specific code** | **~80-100** | **~5** | |

The savings are dramatic: an entire GPipe or 1F1B implementation collapses to constructing a `PipelineStage` and calling `schedule.step()`. The partitioning code (`get_stage`) remains the same in both versions -- the framework does not change how you split the model, only how you execute the pipeline.


### How torchtitan Uses the Same Pattern at Scale

torchtitan's production pipeline parallelism ([torchtitan/distributed/pipeline_parallel.py](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/pipeline_parallel.py)) uses `torch.distributed.pipelining` with additional features:

1. **Automatic stage splitting** via the model's `ModuleDict` structure -- layers are assigned to stages by index range, and unused layers are set to `None` in the model's forward (same pattern shown in the PyTorch docs).

2. **Composability with TP and FSDP** -- the `PipelineStage` wraps a module that may already have TP applied (via `parallelize_module`) and FSDP wrapping. The pipeline schedule operates on top of these other parallelisms.

3. **Schedule selection via config** -- torchtitan supports `ScheduleGPipe`, `Schedule1F1B`, `ScheduleInterleaved1F1B`, and `ScheduleInterleavedZeroBubble` all through config, demonstrating the plug-and-play nature of the schedule API.

```python
# From torchtitan (simplified)
from torch.distributed.pipelining import (
    PipelineStage,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)

# Model is already TP-wrapped and FSDP-wrapped
stage = PipelineStage(model, stage_index, num_stages, device, ...)

# Schedule from config
schedule_cls = {
    "gpipe": ScheduleGPipe,
    "1f1b": Schedule1F1B,
    "interleaved_1f1b": ScheduleInterleaved1F1B,
}[config.schedule]

schedule = schedule_cls(stage, n_microbatches=config.n_microbatches, loss_fn=loss_fn)
```

Our implementation follows the same pattern. The difference is scale: torchtitan runs on hundreds of GPUs with 3D+ parallelism, while our scripts demonstrate the API on 2-4 GPUs for learning.


### Comparison Table: Hand-Written PP vs Framework PP

| Aspect | Hand-Written PP (Part 1) | Framework PP (Part 2) |
|---|---|---|
| **Communication** | Manual `dist.send`/`dist.recv` per microbatch | `PipelineStage` handles all P2P ops |
| **Microbatch splitting** | Manual `input_ids[b*mbs:(b+1)*mbs]` | Framework splits automatically |
| **Activation storage** | Manual lists (`saved_outputs`, `saved_inputs`) | Framework manages buffers |
| **Schedule implementation** | 50-80 lines per schedule | 1 line (`ScheduleGPipe(...)` or `Schedule1F1B(...)`) |
| **Switching schedules** | Rewrite entire schedule function | Change one class name |
| **Model code** | Same (flat `nn.ModuleList`, `get_stage`) | Same (flat `nn.ModuleList`, `get_stage`) |
| **Partitioning** | Same (`meta` device, slice, `to_empty`) | Same (`meta` device, slice, `to_empty`) |
| **Shape requirement** | Implicit (you write the right buffer sizes) | Explicit (`input_args`/`output_args` at construction) |
| **Error mode** | Deadlock if send/recv order is wrong | Framework prevents protocol mismatches |
| **Adding new schedules** | Write from scratch | Extend `PipelineScheduleSingle` or `PipelineScheduleMulti` |
| **Composability with TP/FSDP** | Manual integration | Native support (stage module can be TP/FSDP wrapped) |


### Running the Code

```bash
cd pipeline-parallelism/src

# Framework PP with GPipe schedule: 4 GPUs
torchrun --standalone --nproc_per_node=4 train_gpt_pp_pipelining.py \
    --config mini --pp-size 4 --schedule gpipe --num-microbatches 4

# Framework PP with 1F1B schedule: 4 GPUs
torchrun --standalone --nproc_per_node=4 train_gpt_pp_pipelining.py \
    --config mini --pp-size 4 --schedule 1f1b --num-microbatches 8

# Hand-written equivalents (Part 1, for comparison)
torchrun --standalone --nproc_per_node=4 train_gpt_pp_gpipe.py \
    --config mini --pp-size 4 --num-microbatches 4
torchrun --standalone --nproc_per_node=4 train_gpt_pp_1f1b.py \
    --config mini --pp-size 4 --num-microbatches 8
```

Both versions produce JSON results in `outputs/`. Loss values and throughput should be comparable -- the same model, same partitioning, same schedule semantics. Differences come only from implementation details (buffer allocation strategy, communication overlap).


### Source Code

| File | Description |
|---|---|
| `train_gpt_pp_naive.py` | Naive schedule, N stages ([Part 1](pipeline-parallelism.md)) |
| `train_gpt_pp_gpipe.py` | GPipe schedule, N stages ([Part 1](pipeline-parallelism.md)) |
| `train_gpt_pp_1f1b.py` | 1F1B schedule, N stages ([Part 1](pipeline-parallelism.md)) |
| `train_gpt_pp_pipelining.py` | `torch.distributed.pipelining` (this article) |

All files under [pipeline-parallelism/src/](pipeline-parallelism/src/).


### References

- [torch.distributed.pipelining documentation (PyTorch 2.7)](https://docs.pytorch.org/docs/2.7/distributed.pipelining.html) -- PipelineStage, schedules, splitting frontend
- [torchtitan: Pipeline Parallelism](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/pipeline_parallel.py) -- production PP with composability
- [PiPPy (PyTorch Pipelines)](https://github.com/pytorch/PiPPy) -- original project (migrated to torch.distributed.pipelining)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) (Huang et al., 2019)
- [PipeDream: General Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377) (Narayanan et al., 2019)
- [Memory-Efficient Pipeline-Parallel DNN Training](https://arxiv.org/abs/2004.09505) (Narayanan et al., 2021) -- PipeDream-Flush / 1F1B
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) (Qi et al., 2024)
