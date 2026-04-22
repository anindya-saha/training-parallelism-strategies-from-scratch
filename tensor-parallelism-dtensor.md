## From Hand-Written TP to PyTorch DTensor: A Declarative Sharding Plan

*This is Part 2 of a four-part series on model parallelism. [Part 1](tensor-parallelism-blog.md) builds Tensor Parallelism from scratch using custom autograd primitives. Parts [3](sequence-parallelism-blog.md) and [4](sequence-parallelism-dtensor.md) extend the approach with Sequence Parallelism.*

In [Part 1](tensor-parallelism-blog.md) we built Tensor Parallelism from scratch: `_CopyToParallelRegion`, `_ReduceFromParallelRegion`, `ColumnParallelLinear`, `RowParallelLinear`, and `TPAttention` -- custom autograd functions and modules that fuse the model architecture with the parallelism strategy. The implementation works, but the model cannot run on a single GPU. Changing the TP degree or composing with FSDP requires rewriting module definitions.

This article translates that hand-written implementation to PyTorch's `torch.distributed.tensor.parallel` API. The result is [model_gpt_tp_dtensor.py](tensor-parallelism/src/model_gpt_tp_dtensor.py): a plain `nn.Module` with standard `nn.Linear` layers, parallelized entirely by a single `apply_tp()` function. PyTorch handles weight partitioning, communication insertion, and backward gradient routing.


### The Model: Plain nn.Module

The DTensor approach separates model definition from parallelism strategy. The model code is identical to the single-GPU baseline -- no imports from `torch.distributed`, no `ColumnParallelLinear`, no `n_heads_local`:

```python
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, bias=False):
        super().__init__()
        self.n_heads = n_heads  # global, not local
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

Compare this with the hand-written `TPAttention` from Part 1:

| | Hand-written (`model_gpt_tp.py`) | DTensor (`model_gpt_tp_dtensor.py`) |
|---|---|---|
| Linear layers | `ColumnParallelLinear`, `RowParallelLinear` | `nn.Linear` |
| Head count | `self.n_heads_local = n_heads // ws` | `self.n_heads = n_heads` (global) |
| View reshape | `.view(B, T, self.n_heads_local, self.d_head)` | `.view(B, T, self.n_heads, self.d_head)` |
| Attention impl | Manual `Q @ K.T`, mask, softmax | `F.scaled_dot_product_attention` |
| Communication | `_CopyToParallelRegion`, `_ReduceFromParallelRegion` | None in model code |

The same applies to `FFN`, `TransformerBlock`, and `GPT` -- all standard `nn.Module` with no distribution logic.


### The Sharding Plan: `apply_tp()`

The entire parallelism implementation is a single function:

```python
def apply_tp(model: GPT, mesh) -> GPT:
    for block in model.blocks:
        block_plan = {
            "attn": PrepareModuleInput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Replicate(),),
            ),
            "attn.W_q": ColwiseParallel(use_local_output=False),
            "attn.W_k": ColwiseParallel(use_local_output=False),
            "attn.W_v": ColwiseParallel(use_local_output=False),
            "attn.W_o": RowwiseParallel(),
            "ffn": PrepareModuleInput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Replicate(),),
            ),
            "ffn.W1": ColwiseParallel(),
            "ffn.W2": RowwiseParallel(),
        }
        parallelize_module(block, mesh, block_plan)

    parallelize_module(
        model, mesh,
        {"lm_head": ColwiseParallel(use_local_output=False)},
    )
    return model
```

Each entry in the plan maps a module path (e.g., `"attn.W_q"`) to a `ParallelStyle` that declares how that module should be sharded. `parallelize_module()` does the rest:

1. Swaps parameters to DTensors with the specified sharding.
2. Registers forward pre-hooks for input redistribution.
3. Registers forward post-hooks for output redistribution.
4. The backward pass is derived automatically from the forward DTensor operations.

Reading the plan entry by entry:

| Plan entry | ParallelStyle | What it replaces from Part 1 |
|---|---|---|
| `"attn.W_q"`: `ColwiseParallel(use_local_output=False)` | Shard weight on output dim (`Shard(0)`). Input: `Replicate()` (implicit). Output: `Shard(-1)` DTensor. | `ColumnParallelLinear` + `_CopyToParallelRegion` |
| `"attn.W_k"`: `ColwiseParallel(use_local_output=False)` | Same as W_q | Same as W_q |
| `"attn.W_v"`: `ColwiseParallel(use_local_output=False)` | Same as W_q | Same as W_q |
| `"attn.W_o"`: `RowwiseParallel()` | Shard weight on input dim (`Shard(1)`). All-reduce output to `Replicate()`. | `RowParallelLinear` + `_ReduceFromParallelRegion` |
| `"ffn.W1"`: `ColwiseParallel()` | Shard output dim. Local output (plain tensor). | `ColumnParallelLinear` |
| `"ffn.W2"`: `RowwiseParallel()` | Shard input dim. All-reduce output. | `RowParallelLinear` + `_ReduceFromParallelRegion` |
| `"lm_head"`: `ColwiseParallel(use_local_output=False)` | Shard vocab dim. DTensor output for `loss_parallel()`. | `ColumnParallelLinear` + `_AllGatherFromParallelRegion` |

The `PrepareModuleInput` entries on `attn` and `ffn` annotate the module inputs as `Replicate()` DTensors. In vanilla TP (no Sequence Parallelism), inputs are already replicated, so these are effectively a type annotation -- they ensure DTensor knows the input layout for correct redistribution in the backward pass.


### `use_local_output=False` and View Operations

Notice that W_q, W_k, W_v use `use_local_output=False` while W1 does not. This is not arbitrary.

After `ColwiseParallel` shards W_q column-wise, the output activation has shape `(B, T, d_model // N)` on each GPU -- the last dimension is sharded. The model code then does:

```python
Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
```

This `.view()` uses the *global* `self.n_heads`. If the output were a plain local tensor of shape `(B, T, d_model // N)`, this view would fail -- the local tensor has `d_model // N` elements in the last dimension, but the view expects `n_heads * d_head = d_model` elements.

With `use_local_output=False`, the output stays as a DTensor that knows its *global* shape is `(B, T, d_model)` with placement `Shard(-1)`. DTensor's view operation correctly maps the shard onto the head dimension, producing a DTensor of global shape `(B, T, n_heads, d_head)` sharded on the heads dim. This is why the model stores `self.n_heads` (global) rather than `n_heads // world_size` (local).

The FFN does not have this issue: W1's output feeds directly into `F.gelu()` (element-wise, no reshape), so `use_local_output=True` (the default) works fine. The local shard of shape `(B, T, d_ff // N)` passes through GeLU unchanged and enters W2 which expects a sharded input.

```
ColwiseParallel output on W_q
  |
  +-- use_local_output=True:   local Tensor (B, T, d_model//N)
  |                             .view(B, T, n_heads, d_head) --> FAILS
  |                             (d_model//N != n_heads * d_head)
  |
  +-- use_local_output=False:  DTensor Shard(-1), global shape (B, T, d_model)
                                .view(B, T, n_heads, d_head) --> OK
                                DTensor maps shard onto head dimension
```


### Loss Parallel

In the hand-written model, computing cross-entropy requires gathering vocab-sharded logits:

```python
def tp_cross_entropy(logits_local, labels, vocab_size):
    full = _AllGatherFromParallelRegion.apply(logits_local)
    return F.cross_entropy(full.view(-1, vocab_size), labels.view(-1))
```

This materializes the full `(batch*seq, vocab)` tensor on every GPU. For a 32K vocabulary with batch 8 and sequence 256, that is 256 MB of redundant memory.

DTensor's `loss_parallel()` avoids this. When the LM head uses `ColwiseParallel(use_local_output=False)`, the logits remain as a DTensor sharded on the vocabulary dimension. Inside `loss_parallel()`, `F.cross_entropy` computes the loss without gathering: each GPU computes its local contribution to the log-sum-exp denominator, a small all-reduce synchronizes the denominator, and the final loss is computed from the local shard.

```python
pred = model(input_ids)
with loss_parallel():
    loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
    loss.backward()
```

The backward computation must also happen within the `loss_parallel()` context.

**A practical detail: `foreach=False` on Adam.** With pure TP (no FSDP), the model has a mix of DTensor parameters (parallelized layers) and plain `torch.Tensor` parameters (LayerNorm, embeddings). Adam's fused `_foreach` operations require all parameters to be the same tensor type. Setting `foreach=False` tells Adam to update each parameter individually, avoiding the type mismatch:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
```


### Mapping Table: Hand Primitives to DTensor

| Hand-written primitive | What it does | DTensor equivalent |
|---|---|---|
| `_CopyToParallelRegion` | identity fwd, all-reduce bwd | Implicit in `ColwiseParallel(input_layouts=Replicate())` |
| `_ReduceFromParallelRegion` | all-reduce fwd, identity bwd | Implicit in `RowwiseParallel(output_layouts=Replicate())` |
| `_ScatterToParallelRegion` | scatter fwd, all-gather bwd | Not needed -- cancellation handled automatically |
| `_AllGatherFromParallelRegion` | all-gather fwd, scatter bwd | `loss_parallel()` or `use_local_output=False` |
| `ColumnParallelLinear` | shard weight on output dim + `_CopyToParallelRegion` | `ColwiseParallel()` on `nn.Linear` |
| `RowParallelLinear` | shard weight on input dim + `_ReduceFromParallelRegion` | `RowwiseParallel()` on `nn.Linear` |
| `tp_cross_entropy` | all-gather logits + cross-entropy | `loss_parallel()` context manager |


### Code Eliminated

```
Component                        Hand-written    DTensor
-------------------------------------------------------
_CopyToParallelRegion                 7 lines    0 lines
_ReduceFromParallelRegion             7 lines    0 lines
ColumnParallelLinear                 15 lines    0 lines
RowParallelLinear                    18 lines    0 lines
TPAttention (with n_heads_local)     30 lines    0 lines
_AllGatherFromParallelRegion         12 lines    0 lines
tp_cross_entropy                      3 lines    0 lines
-------------------------------------------------------
Subtotal (parallelism plumbing)     ~92 lines    0 lines
apply_tp() sharding plan              0 lines  ~30 lines
-------------------------------------------------------
Net reduction                                   ~62 lines
```

More importantly, the model is now reusable: the same `GPT` class runs on a single GPU (skip `apply_tp`), with TP (call `apply_tp`), or with TP+FSDP (call `apply_tp` then `fully_shard`).


### Equivalence Tests

Both approaches are verified against the same reference matrices. From the `tensor-parallelism/` directory:

```bash
# Hand-written primitives
torchrun --nproc_per_node=2 src/test_tp_primitives.py

# DTensor primitives
torchrun --nproc_per_node=2 src/test_tp_primitives_dtensor.py
```

Both scripts test three cases:

1. **Column-parallel only:** verify local output matches the corresponding column shard of `X * W1`.
2. **Row-parallel only:** verify all-reduced output matches `X * W2`.
3. **Column then row:** verify fused output matches `X * W1 * W2`.

A subtlety: `nn.Linear` implements `y = x @ weight.T`, so the DTensor tests set `linear.weight` to the transpose of the reference matrix to match the same numeric `X * W` as the hand-written path.


### Running and Comparing

```bash
# Hand-written TP
torchrun --nproc_per_node=2 src/model_gpt_tp.py

# DTensor TP
torchrun --nproc_per_node=2 src/model_gpt_tp_dtensor.py

# With more GPUs
torchrun --nproc_per_node=4 src/model_gpt_tp_dtensor.py --n-heads 8
```

Both produce JSON results in `outputs/`. Loss values will differ slightly due to different RNG paths and `F.scaled_dot_product_attention` (fused kernel) vs manual `Q @ K.T` (explicit matmul). The parameter counts, memory profiles, and communication patterns are equivalent.


### Source Code

| File | Description |
|---|---|
| [model_gpt_tp.py](tensor-parallelism/src/model_gpt_tp.py) | Hand-written TP GPT ([Part 1](tensor-parallelism-blog.md)) |
| [model_gpt_tp_dtensor.py](tensor-parallelism/src/model_gpt_tp_dtensor.py) | DTensor TP GPT (this article) |
| [model_gpt.py](tensor-parallelism/src/model_gpt.py) | Single-GPU baseline |
| [test_tp_primitives.py](tensor-parallelism/src/test_tp_primitives.py) | Hand-written equivalence tests |
| [test_tp_primitives_dtensor.py](tensor-parallelism/src/test_tp_primitives_dtensor.py) | DTensor equivalence tests |


### What's Next

Tensor Parallelism splits weight matrices across GPUs, reducing per-GPU parameter memory. But the operations *between* the parallelized layers -- LayerNorm, Dropout, residual connections -- still operate on the full `(B, S, h)` tensor, replicated identically on every GPU. This wastes both memory and compute.

[Part 3: Sequence Parallelism from Scratch](sequence-parallelism-blog.md) addresses this by sharding along the sequence dimension between the TP regions. The trick: all-reduce is internally reduce-scatter + all-gather. SP splits these apart and inserts useful computation between them -- saving 30-50% of activation memory with zero extra communication.

[Part 4: From Hand-Written TP+SP to DTensor](sequence-parallelism-dtensor.md) applies the same DTensor translation to the combined TP+SP approach, revealing additional subtleties about `output_layouts`, `use_local_output`, and the DTensor consistency rule for residual connections.


### References

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," [arXiv:1909.08053](https://arxiv.org/abs/1909.08053), 2019.
- PyTorch, "Large Scale Transformer model training with Tensor Parallel (TP)," [docs.pytorch.org/tutorials/intermediate/TP_tutorial.html](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html).
- PyTorch, "Tensor Parallel APIs," [docs.pytorch.org/docs/stable/distributed.tensor.parallel.html](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html).
