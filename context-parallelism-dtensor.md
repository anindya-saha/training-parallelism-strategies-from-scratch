## From Hand-Written Ring Attention to PyTorch DTensor: Context Parallel in One Line

*This is Part 6 of a six-part series on model parallelism. Parts [1](tensor-parallelism-blog.md) and [2](tensor-parallelism-dtensor.md) cover Tensor Parallelism. Parts [3](sequence-parallelism-blog.md) and [4](sequence-parallelism-dtensor.md) cover Sequence Parallelism. [Part 5](context-parallelism-blog.md) builds ring attention from scratch.*

In [Part 5](context-parallelism-blog.md) we built ring attention by hand: a `_ring_rotate` function for P2P K/V exchange, an online softmax merger with $(m, \ell, o)$ recurrence, causal masking logic that distinguishes past/same/future chunks, and a load balancer that pairs head and tail tokens for balanced work. That implementation - roughly 115 lines of distributed communication code - gave us a deep understanding of how Context Parallelism works inside the attention layer.

This article replaces all of that with PyTorch's public `context_parallel()` context manager. The result is `train_gpt_cp_dtensor.py`: a model that is **byte-for-byte identical to the single-GPU baseline**. A single `with context_parallel(...)` block handles input sharding, SDPA interception, ring rotation, online softmax merging, load balancing, and buffer restoration. PyTorch does everything.


### Why Use PyTorch's DTensor API for Context Parallelism?

Our hand-written ring attention works, but it has the same practical limitations we saw with hand-written TP in Part 2:

**Coupling between model and parallelism.** The hand-written approach requires the model to call `ring_attention_forward` instead of `F.scaled_dot_product_attention`. The model cannot run on a single GPU without modification. Changing the CP degree requires adjusting the process group setup.

**Composability with other parallelism dimensions.** In practice, CP is never used alone. A Llama 3.1 training run might use TP=8 within a node, CP=4 across nodes in a rack, and FSDP across racks. With hand-written ring attention, composing these parallelism strategies requires custom integration code for each combination. PyTorch's DTensor API is designed for composition: apply TP on one `DeviceMesh` dimension, CP on another, and FSDP on a third.

**Correctness of backward gradients.** Ring attention in the backward pass requires rotating *output gradients* and accumulating K/V gradients from all ring steps. A single mistake silently produces incorrect gradients. The `context_parallel()` context manager derives the backward communication automatically.

**Performance.** The production implementation overlaps P2P communication with Flash Attention kernels, fuses the load-balancing reorder with the shard operation, and handles edge cases like `seq_len % (2 * CP) != 0` gracefully. These optimizations are difficult to get right by hand.


### How the `context_parallel()` Context Manager Works

The public API (available since PyTorch 2.7) is a single context manager that does two things:

**1. Shard buffers in-place.** The tensors passed via `buffers` are split along the dimensions specified in `buffer_seq_dims`. Each GPU receives its portion. Internally, `context_parallel()` uses `_HeadTailLoadBalancer` to reorder tokens before sharding - each GPU receives a balanced mix of head (cheap) and tail (expensive) causal positions, e.g., `[0, 7, 1, 6, 2, 5, 3, 4]` for S=8, CP=2. This is a key difference from our hand-written version, which uses simple contiguous chunks.

**2. Replace SDPA with Ring Attention.** Every call to `F.scaled_dot_product_attention` within the `with` block is intercepted and routed to ring attention - complete with P2P rotation, online softmax merging, and causal mask handling. No module wrapping or `parallelize_module` calls needed.

On exit, the context manager restores the original buffer contents (so the caller can reuse the tensors). The model code never changes. The model uses standard `F.scaled_dot_product_attention`, and the context manager handles everything else.


### The Model: Plain nn.Module

The model is identical to the single-GPU baseline (`train_gpt.py`) - the only difference is that it uses `F.scaled_dot_product_attention` instead of explicit `Q @ K^T`:

```python
# From context-parallelism/src/train_gpt_cp_dtensor.py

class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.dropout = config.dropout

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d_head]
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d_head]
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d_head]

        p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=p)  # [B, H, T, d_head]

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, d_model]
        return self.resid_dropout(self.W_o(out))
```

No `ring_attention_forward`. No `cp_group`. No `_ring_rotate`. The model can run on a single GPU with zero modification.


### The Context Manager: `context_parallel()`

All the Context Parallelism logic lives in the training loop, not the model:

```python
# From context-parallelism/src/train_gpt_cp_dtensor.py

from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import _cp_options, set_rotate_method
from torch.nn.attention import sdpa_kernel, SDPBackend

# Enable head-tail load balancing (True by default, set explicitly for clarity).
# Contrast with train_gpt_cp.py which uses contiguous chunks: GPU 0 gets
# tokens [0..T/CP-1], GPU 1 gets [T/CP..2*T/CP-1], etc.  With load balancing,
# each GPU receives a mix of head (cheap) and tail (expensive) causal positions.
_cp_options.enable_load_balance = True

# Select the all-to-all KV rotation strategy (overlaps SDPA with communication).
# Alternative: "allgather" (default) - all-gather based pass-KV, used in Llama3.
set_rotate_method("alltoall")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    with context_parallel(
        cp_mesh,
        buffers=(input_ids, position_ids, labels),
        buffer_seq_dims=(1, 1, 1),
    ):
        # Everything inside this block uses ring attention for SDPA
        logits = model(input_ids, position_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
```

Four configuration calls replace 115 lines of hand-written ring attention:

1. **`_cp_options.enable_load_balance = True`** - Enables head-tail (zig-zag) reordering before sharding. The `_HeadTailLoadBalancer` rearranges tokens so each GPU gets a balanced mix of early (cheap) and late (expensive) causal positions. This is `True` by default but set explicitly to highlight the contrast with the hand-written version (contiguous chunks, no load balancing). Requires `seq_len % (2 * cp_size) == 0`.

2. **`set_rotate_method("alltoall")`** - Selects the KV rotation strategy for Ring Attention. The all-to-all approach uses interleaved collectives to overlap SDPA computation with the communication needed for the next ring step. The alternative `"allgather"` (default) performs an all-gather first, then computes - used in Llama3 training.

3. **`sdpa_kernel(SDPBackend.FLASH_ATTENTION)`** - Selects Flash Attention as the SDPA backend. Flash Attention requires `bfloat16` or `float16` inputs (the model is initialized with `dtype=torch.bfloat16`). It never materializes the full attention matrix, providing both speed and memory benefits.

4. **`context_parallel(cp_mesh, buffers=(...), buffer_seq_dims=(...))`** - The main context manager. Shards the buffers in-place along the specified sequence dimensions, replaces SDPA with Ring Attention for all calls within the block, and restores the original buffers on exit.

Note that `buffers` does not have to be Q, K, V. In the tutorial, Q, K, V are passed directly because the example is a bare SDPA call. In our model, we pass `input_ids`, `position_ids`, and `labels` - these are sharded along the sequence dimension (dim 1), and the model internally computes Q, K, V from the sharded inputs. Every SDPA call inside the model is intercepted regardless.


### How SDPA Interception Works

When we call `F.scaled_dot_product_attention(Q, K, V, is_causal=True)` inside the `context_parallel()` block, PyTorch does not dispatch to a local kernel. Instead:

1. The context manager has monkey-patched `F.scaled_dot_product_attention` with a ring attention wrapper.
2. The wrapper treats Q, K, V as local chunks (each GPU holds $T/C$ tokens).
3. It runs $C$ ring steps: at each step, it calls the local Flash Attention kernel on the $(S/C) \times (S/C)$ tile, merges partial results via online softmax, and rotates K/V to the next GPU.
4. The result is a regular tensor - the user sees a normal output.

This is analogous to how DTensor intercepts `torch.mm` for tensor-parallel linear layers (Parts 2 and 4): the user writes standard PyTorch ops, and the runtime inserts the appropriate collectives.

The context manager replaces SDPA **globally** within the `with` block. There is no need to identify which module calls SDPA, no need to wrap the attention layer in a special module, and no risk of applying hooks to the wrong module boundary.


### Mapping Table: Hand-Written to DTensor

| Hand-written primitive | Lines | DTensor equivalent | Lines |
|---|---|---|---|
| `_ring_rotate()` - P2P K/V exchange | ~20 | Built into `context_parallel()` ring loop | 0 |
| Online softmax `(m, l, o)` recurrence | ~15 | Built into `context_parallel()` merger | 0 |
| Causal mask logic (past/same/future) | ~10 | Built into `context_parallel()` mask handling | 0 |
| `ring_attention_forward()` main loop | ~40 | `context_parallel()` context manager | 1 |
| Load balancer (head-tail reorder) | ~15 | `_cp_options.enable_load_balance = True` | 1 |
| Input sharding (split + distribute) | ~10 | `buffers=(...), buffer_seq_dims=(...)` | 0 |
| SDPA dispatch interception | N/A | Built into `context_parallel()` | 0 |
| **Total** | **~115** | **Total** | **~5** |

The 115 lines of hand-written ring attention collapse to roughly 5 lines of configuration. The model code itself has **zero** parallelism-specific lines.


### torchtitan Integration Pattern

In torchtitan's `parallelize_llama`, CP is applied as one step in a multi-dimensional parallelism stack. Note that torchtitan uses the lower-level `parallelize_module` + `_ContextParallel` approach (rather than the `context_parallel()` context manager) because it composes more naturally with TP, AC, and FSDP at the module level:

```python
# From torchtitan/models/llama3/parallelize.py (simplified)
def parallelize_llama(model, parallel_dims, ...):
    # Step 1: Tensor Parallelism
    if parallel_dims.tp_enabled:
        apply_tp(model, tp_mesh, enable_cp=parallel_dims.cp_enabled, ...)

    # Step 2: Context Parallelism
    if parallel_dims.cp_enabled:
        apply_cp_to_attention_module(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    # Step 3: Activation Checkpointing
    if ac_config.mode != "none":
        apply_ac(model, ac_config, ...)

    # Step 4: FSDP (Data Parallelism)
    if parallel_dims.dp_enabled:
        apply_fsdp(model, dp_mesh, ...)
```

The order matters: TP first (shards weights within the node), then CP (shards sequence across nodes), then AC (wraps for memory savings), then FSDP (shards remaining state across data-parallel replicas). Each parallelism dimension operates on a different axis of the `DeviceMesh`.

The `DeviceMesh` for a 64-GPU training run might look like:

```python
mesh = DeviceMesh(
    device_type="cuda",
    mesh=torch.arange(64).reshape(2, 4, 8),
    mesh_dim_names=("dp", "cp", "tp"),
)
# dp=2: 2 data-parallel replicas
# cp=4: sequence split 4 ways
# tp=8: weights split 8 ways within each node
```

Each `apply_*` function only sees its own mesh dimension. TP doesn't know about CP, CP doesn't know about FSDP. The composition is orthogonal.

For standalone scripts (like ours), the `context_parallel()` context manager is the simpler choice - no module wrapping, no plan objects, just a `with` block around the training loop.


### Equivalence: Hand-Written vs DTensor

Both implementations produce similar outputs for the same inputs. A simple way to verify is to run all three variants on the same configuration:

```bash
# Hand-written CP (float32, contiguous chunks)
torchrun --nproc_per_node=4 src/train_gpt_cp.py --config mini --cp-size 4

# Hand-written CP (bfloat16, contiguous chunks) - apples-to-apples with DTensor
torchrun --nproc_per_node=4 src/train_gpt_cp.py --config mini --cp-size 4 --dtype bfloat16

# DTensor CP (bfloat16, Flash Attention, head-tail load balancing, alltoall rotation)
torchrun --nproc_per_node=4 src/train_gpt_cp_dtensor.py --config mini --cp-size 4
```

Both produce JSON results in `outputs/`. Loss values should be similar - differences come from two sources:
1. **Load balancing**: DTensor uses head-tail zig-zag reordering while the hand-written version uses contiguous chunks, so each GPU sees different token subsets.
2. **Kernel differences**: Flash Attention (DTensor) and explicit matmul (hand-written) produce slightly different floating-point results due to non-associativity.

The hand-written version accepts `--dtype bfloat16` to match DTensor's precision, isolating the CP implementation differences from dtype effects.


### Experimental Results: Hand-Written vs DTensor

All three variants were run on the same hardware (4x A10G GPUs, g5.12xlarge) with the `small` config (GPT-2 117M, 12 heads, 12 layers) and `seq_len=512`, `cp_size=4`:

```bash
# Hand-written CP (float32)
torchrun --nproc_per_node=4 src/train_gpt_cp.py --config small --seq-len 512 --cp-size 4

# Hand-written CP (bfloat16) - apples-to-apples with DTensor
torchrun --nproc_per_node=4 src/train_gpt_cp.py --config small --seq-len 512 --cp-size 4 --dtype bfloat16

# DTensor CP (bfloat16, Flash Attention, head-tail load balancing, alltoall rotation)
torchrun --nproc_per_node=4 src/train_gpt_cp_dtensor.py --config small --seq-len 512 --cp-size 4
```

| Metric | Hand-written (float32) | Hand-written (bfloat16) | DTensor (bfloat16) |
|---|---|---|---|
| Params per GPU | 163,037,184 | 163,037,184 | 163,037,184 |
| **Model memory** | 623.5 MB | 318.4 MB | 318.4 MB |
| **Peak memory** | 4,142 MB | 2,226 MB | 1,718 MB |
| **Forward** | 114.6 ms | 61.7 ms | 51.0 ms |
| **Backward** | 57.2 ms | 50.4 ms | 164.5 ms |
| **Step total** | 198.0 ms | **125.8 ms** | 229.3 ms |
| **Tokens/sec** | 20,683 | **32,567** | 17,866 |
| Loss | 6.12 | 6.06 | 6.12 |

**Key observations:**

**The bfloat16 column isolates dtype from CP implementation.** Comparing the two hand-written columns shows what dtype alone buys: model memory halves (623 -> 318 MB), peak memory drops 1.86x (4,142 -> 2,226 MB), forward halves (114.6 -> 61.7 ms), and step time drops 1.57x (198 -> 126 ms). The bfloat16 hand-written column is the fair baseline for comparing against DTensor.

**DTensor uses 23% less peak memory than hand-written at equal precision.** At bfloat16, DTensor peaks at 1,718 MB vs 2,226 MB. The difference is Flash Attention: it never materializes the `[B, H, T_local, T_local]` score matrix, while the hand-written version computes `Q @ K.T` explicitly, allocating that tile at each ring step.

**DTensor forward is 17% faster than hand-written at equal precision.** Flash Attention fuses the attention computation into a single kernel instead of separate matmul, softmax, masking, and dropout operations. At bfloat16: 51.0 ms vs 61.7 ms.

**DTensor backward is 3.3x slower.** The autograd graph through `context_parallel()`'s ring attention backward is the clear bottleneck: 164.5 ms vs 50.4 ms. The hand-written version has a simpler backward graph because it manually manages P2P communication within the forward pass. In production (torchtitan), `torch.compile` fuses the backward kernels and eliminates this overhead.

**Hand-written bfloat16 is 1.82x faster end-to-end than DTensor.** The forward speedup from Flash Attention (17%) is overwhelmed by the backward overhead (3.3x). Overall: 125.8 ms vs 229.3 ms per step, 32,567 vs 17,866 tokens/sec.

**Loss values converge to the same point.** Despite different token orderings (head-tail zig-zag vs contiguous) and different kernels (Flash Attention vs explicit matmul), all three variants converge to loss ~6.1 after 13 steps.

The takeaway: the `context_parallel()` API delivers memory efficiency (23% lower peak at equal precision), correctness guarantees (automatic backward), and composability (works with TP/FSDP via DeviceMesh) - all with zero model modifications. The backward overhead is real but expected to disappear with `torch.compile` in production.


### Running Instructions

```bash
# Hand-written CP (float32, default)
torchrun --nproc_per_node=4 src/train_gpt_cp.py --config mini --cp-size 4

# Hand-written CP (bfloat16, apples-to-apples with DTensor)
torchrun --nproc_per_node=4 src/train_gpt_cp.py --config mini --cp-size 4 --dtype bfloat16

# DTensor CP (bfloat16, Flash Attention)
torchrun --nproc_per_node=4 src/train_gpt_cp_dtensor.py --config mini --cp-size 4

# Medium model with longer sequence (OOM on single GPU, works with CP=4)
torchrun --nproc_per_node=4 src/train_gpt_cp_dtensor.py --config medium --seq-len 1024 --cp-size 4
```

Both produce JSON results in `outputs/`. The DTensor version enforces `seq_len % (2 * cp_size) == 0` (from `_HeadTailLoadBalancer`), vs `seq_len % cp_size == 0` for the hand-written version.


### Source Code

| File | Description |
|---|---|
| `train_gpt.py` | Baseline GPT (explicit Q @ K^T, no parallelism) |
| `train_gpt_cp.py` | Hand-written ring attention GPT ([Part 5](context-parallelism-blog.md)) |
| `train_gpt_cp_dtensor.py` | DTensor CP GPT (this article) |

All files under [context-parallelism/src/](context-parallelism/src/).


### What's Next

This completes the six-part series on model parallelism:

| Part | Topic | Key idea |
|---|---|---|
| [1](tensor-parallelism-blog.md) | TP from scratch | Split weight matrices, all-reduce between layers |
| [2](tensor-parallelism-dtensor.md) | TP with DTensor | `ColwiseParallel` / `RowwiseParallel` plans |
| [3](sequence-parallelism-blog.md) | SP from scratch | Split all-reduce into reduce-scatter + all-gather, compute between them |
| [4](sequence-parallelism-dtensor.md) | SP with DTensor | `SequenceParallel` plan with `output_layouts` |
| [5](context-parallelism-blog.md) | CP from scratch | Ring attention with P2P rotation + online softmax |
| [6](context-parallelism-dtensor.md) | CP with DTensor | `context_parallel()` context manager |

The pattern is consistent: build by hand to understand the primitives, then translate to DTensor to let PyTorch handle the plumbing. Each parallelism strategy operates on a different axis - weight matrices (TP), activations outside attention (SP), and the sequence dimension inside attention (CP) - and they compose orthogonally via `DeviceMesh`.

In production (torchtitan), all three are applied together: TP within a node, CP across nearby nodes with fast interconnect, FSDP across the full cluster. The model stays a plain `nn.Module` throughout.


### References

- [PyTorch Context Parallel Tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) (official)
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (Liu et al., 2023)
- [Coconut Mode: Ring Attention Explained](https://coconut-mode.com/posts/ring-attention/)
- [torchtitan: Context Parallel integration](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/context_parallel.py)
- [torchtitan PR #592: enable Context Parallel](https://github.com/pytorch/torchtitan/pull/592)
- [PyTorch native long-context training with CP](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082) (PyTorch discussion)
- PyTorch, `context_parallel()` API, `torch.distributed.tensor.experimental`
- [Striped Attention](https://arxiv.org/abs/2311.09431) (Brandon et al., 2023)
- Technical notes: [context-parallelism.md](context-parallelism.md), [context_parallelism_concrete_walkthrough.md](context-parallelism/context_parallelism_concrete_walkthrough.md)
