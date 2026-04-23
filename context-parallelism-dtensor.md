## From Hand-Written Ring Attention to PyTorch DTensor: Context Parallel in One Line

*This is Part 6 of a six-part series on model parallelism. Parts [1](tensor-parallelism-blog.md) and [2](tensor-parallelism-dtensor.md) cover Tensor Parallelism. Parts [3](sequence-parallelism-blog.md) and [4](sequence-parallelism-dtensor.md) cover Sequence Parallelism. [Part 5](context-parallelism-blog.md) builds ring attention from scratch.*

In [Part 5](context-parallelism-blog.md) we built ring attention by hand: a `_ring_rotate` function for P2P K/V exchange, an online softmax merger with $(m, \ell, o)$ recurrence, causal masking logic that distinguishes past/same/future chunks, and a load balancer that pairs head and tail tokens for balanced work. That implementation -- roughly 115 lines of distributed communication code -- gave us a deep understanding of how Context Parallelism works inside the attention layer.

This article translates that hand-written implementation to PyTorch's `_ContextParallel` API. The result is `model_gpt_cp_dtensor.py`: a plain `nn.Module` that uses standard `F.scaled_dot_product_attention`, parallelized entirely by a single `apply_cp()` function. PyTorch handles the ring rotation, the online softmax merging, and the load balancing transparently.


### Why Use PyTorch's DTensor API for Context Parallelism?

Our hand-written ring attention works, but it has the same practical limitations we saw with hand-written TP in Part 2:

**Coupling between model and parallelism.** The hand-written approach requires the model to call `ring_attention_forward` instead of `F.scaled_dot_product_attention`. The model cannot run on a single GPU without modification. Changing the CP degree requires adjusting the process group setup.

**Composability with other parallelism dimensions.** In practice, CP is never used alone. A Llama 3.1 training run might use TP=8 within a node, CP=4 across nodes in a rack, and FSDP across racks. With hand-written ring attention, composing these parallelism strategies requires custom integration code for each combination. PyTorch's DTensor API is designed for composition: apply TP on one `DeviceMesh` dimension, CP on another, and FSDP on a third.

**Correctness of backward gradients.** Ring attention in the backward pass requires rotating *output gradients* and accumulating K/V gradients from all ring steps. A single mistake silently produces incorrect gradients. The DTensor `_ContextParallel` plan derives the backward communication automatically.

**Performance.** The production implementation overlaps P2P communication with Flash Attention kernels, fuses the load-balancing reorder with the shard operation, and handles edge cases like `seq_len % (2 * CP) != 0` gracefully. These optimizations are difficult to get right by hand.


### How DTensor Context Parallel Works

At a high level, DTensor CP operates in three phases:

**Phase 1 -- Sharding.** Before the model forward pass, input tokens `[B, S]` and labels are sharded along the sequence dimension across the CP group. The `_HeadTailLoadBalancer` reorders tokens before sharding to balance causal attention work. This is done once, before the model sees the data.

**Phase 2 -- Interception.** Each attention module is parallelized with a `_ContextParallel` plan. When the model calls `F.scaled_dot_product_attention`, the plan intercepts the call and routes it to the ring attention implementation -- complete with P2P rotation, online softmax merging, and causal mask handling.

**Phase 3 -- Restoration.** After the forward pass (and loss computation), the load balancer restores the original token order. Gradients flow backward through the same ring topology.

The user's model code never changes. The model uses `F.scaled_dot_product_attention`, and the DTensor machinery handles everything else.


### The Model: Plain nn.Module

The DTensor approach separates model definition from parallelism. The model is identical to the single-GPU baseline -- no ring attention imports, no P2P operations, no online softmax accumulators:

<!-- TODO: Code from context-parallelism/src/model_gpt_cp_dtensor.py once built -->

```python
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)
```

Notice: standard `F.scaled_dot_product_attention`. No `ring_attention_forward`. No `cp_group`. No `_ring_rotate`. The model can run on a single GPU with zero modification.


### The Sharding Plan: apply_cp()

All the Context Parallelism logic lives in a single function:

<!-- TODO: Code from context-parallelism/src/model_gpt_cp_dtensor.py once built -->

```python
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _ContextParallel,
    _enable_context_parallel_dispatcher,
    _HeadTailLoadBalancer,
)
from torch.distributed.tensor.parallel import parallelize_module

def apply_cp(model, cp_mesh):
    """Apply Context Parallelism to all attention modules."""
    _enable_context_parallel_dispatcher()

    cp_plan = _ContextParallel(
        seq_dim=1,
        attention_type=_ContextParallel.AttentionType.SDPA,
    )

    for block in model.transformer.h:
        parallelize_module(
            module=block.attn,
            device_mesh=cp_mesh,
            parallelize_plan=cp_plan,
        )
```

Three API calls replace 115 lines of hand-written ring attention:

1. **`_enable_context_parallel_dispatcher()`** -- Installs a DTensor dispatcher that intercepts `F.scaled_dot_product_attention` calls and routes them to the ring attention implementation. Without this call, SDPA runs locally with no inter-GPU communication. (For `FlexAttention`, this dispatcher is not needed -- the interception works differently.)

2. **`_ContextParallel(seq_dim=1)`** -- Creates a parallelization plan that tells PyTorch: "the sequence dimension of Q, K, V is axis 1 (i.e., shape `[B, H, T, d]`), and I want ring attention on it." The `attention_type=SDPA` flag selects the SDPA code path (vs FlexAttention).

3. **`parallelize_module()`** -- Applies the plan to each attention module. After this call, every forward pass through the module automatically shards Q along the sequence dimension, runs the ring rotation + online softmax loop, and assembles the output.


### Input Sharding: _context_parallel_shard()

Before the model forward pass, inputs and labels must be sharded along the sequence dimension:

<!-- TODO: Code from context-parallelism/src/model_gpt_cp_dtensor.py once built -->

```python
def shard_inputs_for_cp(inputs, labels, cp_mesh):
    """Shard inputs and labels along the sequence dimension with load balancing."""
    seq_len = inputs.size(1)
    cp_world_size = cp_mesh.size(0)

    load_balancer = _HeadTailLoadBalancer(
        seq_len, cp_world_size, cp_mesh.device_type
    )

    inputs, labels = _context_parallel_shard(
        mesh=cp_mesh,
        buffers=(inputs, labels),
        seq_dims=(1, 1),
        load_balancer=load_balancer,
    )

    return inputs, labels
```

`_context_parallel_shard` does two things:
1. If a load balancer is provided, it reorders tokens using the head-tail pattern (e.g., `[0, 7, 1, 6, 2, 5, 3, 4]` for S=8, C=2).
2. It splits the reordered sequence into $C$ equal chunks and distributes chunk $r$ to rank $r$.

The result: each GPU receives a balanced mix of head and tail tokens, and `F.scaled_dot_product_attention` in the model forward will be intercepted by the CP dispatcher.


### The Load Balancer: _HeadTailLoadBalancer

In [Part 5](context-parallelism-blog.md), we described the causal masking imbalance and the head-tail reordering solution. PyTorch's `_HeadTailLoadBalancer` implements exactly this:

```python
load_balancer = _HeadTailLoadBalancer(
    seq_len=seq_len,
    world_size=cp_world_size,
    device=device,
)
```

Internally, it generates the zig-zag permutation indices and applies them before sharding. After the forward pass and loss computation, the inverse permutation restores original token order for the backward pass.

The constraint `seq_len % (2 * CP) == 0` is enforced -- if violated, `_HeadTailLoadBalancer` raises an error. In torchtitan, the full constraint accounting for TP is `seq_len % (TP * 2 * CP) == 0`.

An alternative load balancer, `_PTRRLoadBalancer` (Partial Token Round Robin), is used with FlexAttention and `BlockMask`. It provides finer-grained load balancing that works with complex attention patterns beyond simple causal.


### Mapping Table: Hand-Written to DTensor

| Hand-written primitive | Lines | DTensor equivalent | Lines |
|---|---|---|---|
| `_ring_rotate()` -- P2P K/V exchange | ~20 | Built into `_ContextParallel` ring loop | 0 |
| Online softmax `(m, l, o)` recurrence | ~15 | Built into `_ContextParallel` merger | 0 |
| Causal mask logic (past/same/future) | ~10 | Built into `_ContextParallel` mask handling | 0 |
| `ring_attention_forward()` main loop | ~40 | `_ContextParallel(seq_dim=1)` plan | 1 |
| Load balancer (head-tail reorder) | ~15 | `_HeadTailLoadBalancer` | 1 |
| Input sharding (split + distribute) | ~10 | `_context_parallel_shard()` | 1 |
| SDPA dispatch interception | N/A | `_enable_context_parallel_dispatcher()` | 1 |
| **Total** | **~115** | **Total** | **~15** |

The 115 lines of hand-written ring attention collapse to roughly 15 lines of DTensor configuration. The model code itself has zero parallelism-specific lines.


### The SDPA Dispatcher: How Interception Works

The `_enable_context_parallel_dispatcher()` call deserves deeper explanation. When you call `F.scaled_dot_product_attention(Q, K, V, is_causal=True)`, PyTorch dispatches to a kernel (Flash, cuDNN, or math). With the CP dispatcher enabled:

1. PyTorch detects that Q, K, V are DTensors with a `_ContextParallel` placement on the CP mesh dimension.
2. Instead of dispatching to a local kernel, it routes to the ring attention implementation.
3. The ring implementation runs $C$ steps: at each step, it calls the local Flash Attention kernel on the $(S/C) \times (S/C)$ tile, merges via online softmax, and rotates K/V.
4. The result is an ordinary DTensor with the same placement -- the user sees a normal output tensor.

This is analogous to how DTensor intercepts `torch.mm` for tensor-parallel linear layers (Parts 2 and 4): the user writes standard PyTorch ops, and the DTensor runtime inserts the appropriate collectives.


### torchtitan Integration Pattern

In torchtitan's `parallelize_llama`, CP is applied as one step in a multi-dimensional parallelism stack:

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


### Equivalence: Hand-Written vs DTensor

Both implementations should produce identical outputs for the same inputs. A correctness test verifies this:

<!-- TODO: Build context-parallelism/src/test_cp_equivalence.py -->

```python
def test_cp_equivalence():
    """Verify hand-written ring attention matches DTensor CP output."""
    # 1. Run hand-written ring attention
    out_hand = ring_attention_forward(q_local, k_local, v_local, cp_group)

    # 2. Run DTensor CP (same model, same weights, same input)
    apply_cp(model, cp_mesh)
    out_dtensor = model(input_sharded)

    # 3. Compare -- should match within floating-point tolerance
    torch.testing.assert_close(out_hand, out_dtensor, rtol=1e-4, atol=1e-4)
```

The test runs both paths on the same input and weights, comparing outputs element-wise. Differences should be at most floating-point rounding (different kernel orderings may produce slightly different results in bf16, but the f32 accumulators should match closely).


### Running Instructions

<!-- TODO: Replace with actual commands once model_gpt_cp_dtensor.py is built -->

```bash
# Hand-written CP (from Part 5)
torchrun --nproc_per_node=4 context-parallelism/src/model_gpt_cp.py --cp_size 4

# DTensor CP (this article)
torchrun --nproc_per_node=4 context-parallelism/src/model_gpt_cp_dtensor.py --cp_size 4

# Equivalence test
torchrun --nproc_per_node=4 context-parallelism/src/test_cp_equivalence.py
```

Both produce JSON results in `outputs/`. Loss values should be nearly identical -- any differences come from kernel dispatch order and floating-point non-associativity, not from algorithmic differences.


### Source Code

| File | Description |
|---|---|
| `model_gpt_cp.py` | Hand-written ring attention GPT ([Part 5](context-parallelism-blog.md)) |
| `model_gpt_cp_dtensor.py` | DTensor CP GPT (this article) |
| `ring_attention.py` | Standalone ring attention forward function |
| `test_ring_attention.py` | Ring attention correctness test vs SDPA |
| `test_cp_equivalence.py` | Hand-written vs DTensor equivalence test |

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
| [6](context-parallelism-dtensor.md) | CP with DTensor | `_ContextParallel` plan with `_HeadTailLoadBalancer` |

The pattern is consistent: build by hand to understand the primitives, then translate to DTensor to let PyTorch handle the plumbing. Each parallelism strategy operates on a different axis -- weight matrices (TP), activations outside attention (SP), and the sequence dimension inside attention (CP) -- and they compose orthogonally via `DeviceMesh`.

In production (torchtitan), all three are applied together: TP within a node, CP across nearby nodes with fast interconnect, FSDP across the full cluster. The model stays a plain `nn.Module` throughout.


### References

- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (Liu et al., 2023)
- [Coconut Mode: Ring Attention Explained](https://coconut-mode.com/posts/ring-attention/)
- [torchtitan: Context Parallel integration](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/context_parallel.py)
- [torchtitan PR #592: enable Context Parallel](https://github.com/pytorch/torchtitan/pull/592)
- PyTorch, "_ContextParallel API," `torch.distributed.tensor.experimental._attention`
- [Striped Attention](https://arxiv.org/abs/2311.09431) (Brandon et al., 2023)
- Technical notes: [context-parallelism.md](context-parallelism.md), [context_parallelism_concrete_walkthrough.md](context-parallelism/context_parallelism_concrete_walkthrough.md)
