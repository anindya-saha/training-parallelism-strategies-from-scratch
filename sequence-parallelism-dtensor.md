## From Hand-Written TP+SP to PyTorch DTensor: A Practical Translation

*This is Part 4 of a four-part series on model parallelism. Parts [1](tensor-parallelism-blog.md) and [2](tensor-parallelism-dtensor.md) cover Tensor Parallelism. [Part 3](sequence-parallelism-blog.md) builds TP+SP from scratch. This article translates the combined TP+SP to the DTensor API.*

In [Part 3](sequence-parallelism-blog.md) we built Tensor Parallelism + Sequence Parallelism from scratch: custom `torch.autograd.Function` subclasses for communication primitives, manual weight sharding in `ColumnParallelLinear` and `RowParallelLinear`, and explicit region transitions between the TP and SP regions. That implementation -- roughly 250 lines of parallelism plumbing in [model_gpt_tp_sp.py](sequence-parallelism/src/model_gpt_tp_sp.py) -- gave us a deep understanding of how TP+SP works.

This article translates that hand-written implementation to PyTorch's `torch.distributed.tensor.parallel` API. The result is [model_gpt_tp_sp_dtensor.py](sequence-parallelism/src/model_gpt_tp_sp_dtensor.py): a plain `nn.Module` with no parallelism baked into its definition, parallelized entirely by a single `apply_tp_sp()` function that declares a sharding plan. PyTorch handles weight partitioning, communication insertion, and backward gradient routing.

The translation was not trivial. Two bugs along the way revealed fundamental concepts about how DTensor manages tensor layouts and collective communication. We document them here because they teach more about the API than the working code alone.


### Why Use PyTorch's DTensor API for Tensor Parallelism?

Our hand-written TP+SP implementation works, but it has practical limitations that become apparent at scale:

**Coupling between model and parallelism.** In the hand-written approach, `ColumnParallelLinear`, `RowParallelLinear`, and `TPAttention` fuse the model architecture with the parallelism strategy. Changing the TP degree, switching between TP and TP+SP, or adding FSDP on top requires rewriting module definitions. The model cannot run on a single GPU without modification.

**Composability with other parallelism dimensions.** As models scale beyond what TP alone can handle, practitioners combine TP with FSDP (data parallelism) and pipeline parallelism. With the hand-written approach, each combination requires custom integration code. PyTorch's DTensor API is designed for composition: apply TP intra-host on one `DeviceMesh` dimension, then apply FSDP inter-host on another, with zero changes to the model or the TP plan.

**Correctness of backward gradients.** Our custom `torch.autograd.Function` subclasses must manually implement the conjugate communication pattern (all-gather forward / reduce-scatter backward, and vice versa). A single mistake in the backward pass silently produces incorrect gradients. DTensor derives the backward communication automatically from the forward sharding specification.

**Performance.** DTensor can fuse communication with computation, overlap collectives with subsequent operations, and leverage async collective operations -- optimizations that are difficult to implement correctly by hand.

The PyTorch [Tensor Parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) summarizes the design philosophy: "Users would only need to specify how to shard the individual layers and the communications will happen under the hood."


### How PyTorch DTensor Tensor Parallel Works

At a high level, DTensor Tensor Parallel operates in two phases:

**Phase 1: Sharding initialization.** You define a `parallelize_plan` -- a dictionary mapping module FQN (fully qualified name) paths to `ParallelStyle` objects -- and call `parallelize_module()`. Under the hood, this:

- Swaps each targeted module's parameters from plain `torch.Tensor` to `DTensor`, partitioned according to the style (e.g., `Shard(0)` for column-wise, `Shard(1)` for row-wise).
- Registers `forward_pre_hook` functions that annotate and redistribute input tensors to the layout the module expects.
- Registers `forward_post_hook` functions that redistribute output tensors to the layout downstream modules expect, and optionally unwrap DTensors back to plain tensors.

**Phase 2: Runtime forward/backward.** During forward, each parallelized module:

1. Receives input, the pre-hook converts it to the required DTensor layout (triggering collectives like all-gather if needed).
2. Runs the sharded computation (e.g., `nn.Linear` with a sharded weight produces a sharded output).
3. The post-hook redistributes the output (triggering collectives like reduce-scatter if needed).

The backward pass is automatic: DTensor records the forward redistributions in the autograd graph and generates the conjugate collectives for gradient computation.


### The ParallelStyle API

PyTorch provides a set of module-level primitives (`ParallelStyle` subclasses) for configuring the sharding of each layer:

| ParallelStyle | Supported modules | What it does |
|---|---|---|
| `ColwiseParallel` | `nn.Linear`, `nn.Embedding` | Shards weight on the output dimension (`Shard(0)`). Input defaults to `Replicate()`, output defaults to `Shard(-1)`. |
| `RowwiseParallel` | `nn.Linear`, `nn.Embedding` | Shards weight on the input dimension (`Shard(1)`). Input defaults to `Shard(-1)`, output defaults to `Replicate()`. |
| `SequenceParallel` | `nn.LayerNorm`, `nn.Dropout`, `RMSNorm` | Replicates parameters, runs sharded computation on `Shard(1)` (sequence dim) input/output. |
| `PrepareModuleInput` | Any `nn.Module` | Registers a pre-hook that annotates inputs as DTensors with `input_layouts` and redistributes to `desired_input_layouts`. |
| `PrepareModuleOutput` | Any `nn.Module` | Registers a post-hook that annotates outputs and redistributes to desired layouts. |

Each style accepts optional `input_layouts`, `output_layouts`, and `use_local_output` parameters that control the tensor layout at module boundaries and whether the output is unwrapped from DTensor to a plain tensor.

The key insight: **you specify layouts, not collectives.** When you write `PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),))`, you are saying "the input arrives sharded on dim 1, and the module needs it replicated." DTensor infers that an all-gather is required. You never call `dist.all_gather` yourself.


### The Model: Plain nn.Module, No Parallelism Baked In

The entire point of the DTensor approach is separation of concerns. The model code knows nothing about distribution:

```python
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, bias=False, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=p)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)
```

Compare this with the hand-written version, where `TPAttention` uses `ColumnParallelLinear` (which internally calls `_CopyToTPRegion.apply()`), `RowParallelLinear` (which internally calls `_ReduceScatterToSPRegion.apply()`), stores `n_heads_local` instead of global `n_heads`, and requires the caller to all-gather the input before entry. The hand-written module *is* the parallelism code. The DTensor module is just a transformer.

The same holds for `FFN`, `TransformerBlock`, and `GPT`: standard `nn.Linear`, `nn.LayerNorm`, `nn.Dropout`, `nn.Embedding`. No imports from `torch.distributed` anywhere in the model definition.

**A note on `use_local_output=False` and view operations.** When `ColwiseParallel` shards W_q/W_k/W_v column-wise, the output activation is sharded on the last dimension (the `num_heads * d_head` dimension). The subsequent `.view(B, T, n_heads, d_head)` reshape needs to use the *global* `n_heads`, not the local shard size. With `use_local_output=False`, the output remains a DTensor that is aware of its global shape. DTensor automatically maps the shard onto the head dimension during the view operation. This is why `Attention` stores `self.n_heads` (global) rather than `n_heads // world_size` (local) -- unlike the hand-written `TPAttention` which must track `n_heads_local` explicitly.

```
+---------------------------------------------------------------+
|                    What you write                             |
|                                                               |
|   Attention:                           GPT:                   |
|     W_q = nn.Linear(d, d)               tok_emb = Embedding   |
|     W_k = nn.Linear(d, d)               pos_emb = Embedding   |
|     W_v = nn.Linear(d, d)               emb_dropout = Dropout  |
|     W_o = nn.Linear(d, d)               blocks = [...]        |
|                                          lm_head = Linear     |
+---------------------------------------------------------------+
                          |
                   apply_tp_sp(model, mesh)
                          |
                          v
+---------------------------------------------------------------+
|              What PyTorch inserts (via hooks)                 |
|                                                               |
|   forward_pre_hook on each submodule:                         |
|     - Annotate input as Shard(1) DTensor                      |
|     - Redistribute: Shard(1) -> Replicate  [all-gather]       |
|                                                               |
|   Weight partitioning:                                        |
|     - ColwiseParallel: shard weight on dim 0  [Shard(0)]      |
|     - RowwiseParallel: shard weight on dim 1  [Shard(1)]      |
|                                                               |
|   forward_post_hook on each submodule:                        |
|     - Redistribute: partial -> Shard(1)  [reduce-scatter]     |
|     - Optionally unwrap DTensor -> local tensor               |
+---------------------------------------------------------------+
```


### The Sharding Plan: One Function Replaces Everything

The function `apply_tp_sp()` is the entire parallelism implementation. It calls `parallelize_module()` three times: once for embeddings, once per transformer block, and once for the output head.

**Stage 1: Embeddings**

```python
parallelize_module(
    model, mesh,
    {
        "tok_emb": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        "pos_emb": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
    },
)
```

`RowwiseParallel` on `nn.Embedding` shards the embedding table rows across GPUs (`Shard(0)` on the weight). `output_layouts=Shard(1)` means the output is sequence-sharded -- we enter the SP region immediately. `use_local_output=False` keeps the output as a DTensor so downstream modules see the `Shard(1)` placement.

In the hand-written model, the embedding is replicated on all GPUs, and each GPU manually slices its chunk of the sequence:

```python
x = self.tok_emb(input_ids) + self.pos_emb(pos)  # (B, S, h)
start = self.rank * S_local
x = x[:, start : start + S_local, :].contiguous()  # (B, S/N, h)
```

DTensor handles this scatter automatically via the embedding sharding strategy.


**Stage 2: Transformer blocks**

Each block gets the same plan:

```python
block_plan = {
    "norm1": SequenceParallel(),
    "attn": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "attn.W_q": ColwiseParallel(use_local_output=False),
    "attn.W_k": ColwiseParallel(use_local_output=False),
    "attn.W_v": ColwiseParallel(use_local_output=False),
    "attn.W_o": RowwiseParallel(
        output_layouts=Shard(1),
        use_local_output=False,
    ),
    "norm2": SequenceParallel(),
    "ffn": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "ffn.W1": ColwiseParallel(),
    "ffn.W2": RowwiseParallel(
        output_layouts=Shard(1),
        use_local_output=False,
    ),
}
```

Reading this plan from top to bottom traces the same data flow as the hand-written block:

1. `norm1` with `SequenceParallel()` -- LayerNorm operates on `Shard(1)` input, stays in the SP region.
2. `attn` with `PrepareModuleInput(Shard(1) -> Replicate)` -- all-gather reconstructs the full sequence before attention. This replaces `_AllGatherFromSPRegion.apply()`.
3. `W_q, W_k, W_v` with `ColwiseParallel` -- split output dimension across GPUs. The implicit `Replicate()` input layout replaces `_CopyToTPRegion.apply()` inside `ColumnParallelLinear`.
4. `W_o` with `RowwiseParallel(output_layouts=Shard(1))` -- reduce-scatter the partial sums back to the SP region. This replaces `_ReduceScatterToSPRegion.apply()` inside `RowParallelLinear`.
5. The FFN sub-block follows the same pattern.

Note how this plan mirrors the structure from the PyTorch [TP tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html), which applies the same pattern to Llama 2. The Llama attention layer has a `PrepareModuleInput` with `input_layouts=(Shard(1), Replicate())` -- two elements because Llama's attention forward takes `(x, freqs_cis)` as positional arguments. Our `Attention.forward(self, x)` takes only one tensor argument, so our tuple is length-1: `(Shard(1),)`. The tuple length must match the number of positional tensor arguments to the module's `forward` method.


**Stage 3: Output head and loss**

```python
parallelize_module(
    model, mesh,
    {
        "norm_f": SequenceParallel(),
        "lm_head": ColwiseParallel(
            input_layouts=Shard(1),
            use_local_output=False,
        ),
    },
)
```

The final LayerNorm stays in the SP region. `lm_head` with `ColwiseParallel(input_layouts=Shard(1))` all-gathers the input and produces vocab-sharded logits. With `use_local_output=False`, the logits remain as a DTensor so that `loss_parallel()` can compute cross-entropy without materializing the full `(batch*seq, vocab)` tensor on any single GPU.


### Loss Parallel: Efficient Cross-Entropy on Sharded Logits

In the hand-written model, computing cross-entropy loss requires gathering the vocab-sharded logits from all GPUs:

```python
class _AllGatherFromParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        gathered = [torch.zeros_like(x) for _ in range(ws)]
        dist.all_gather(gathered, x.contiguous())
        return torch.cat(gathered, dim=-1)
    # ... backward scatters gradients back

def tp_cross_entropy(logits_local, labels, vocab_size):
    full = _AllGatherFromParallelRegion.apply(logits_local)
    return F.cross_entropy(full.view(-1, vocab_size), labels.view(-1))
```

This materializes the full `(batch*seq, vocab)` logits tensor on every GPU -- for a 32K vocabulary, that is `batch * seq * 32K * 4` bytes per GPU. At batch=4, seq=2048, this is 1 GB of redundant memory.

The DTensor `loss_parallel()` context manager avoids this entirely. When logits are a DTensor sharded on the vocabulary dimension (`Shard(-1)` from `ColwiseParallel`), `F.cross_entropy` inside `loss_parallel()` computes the loss without gathering: each GPU computes its local contribution to the log-sum-exp denominator, a small all-reduce synchronizes the denominator, and the final loss is computed from the local shard. Only a scalar all-reduce is needed instead of a full all-gather of the logits.

```python
pred = model(input_ids)
with loss_parallel():
    loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
    loss.backward()
```

The backward computation must also happen within the `loss_parallel()` context so that gradients flow correctly through the sharded cross-entropy.


### Mapping Table: Hand Primitives to DTensor Styles

| Hand-written primitive | What it does | DTensor equivalent |
|---|---|---|
| `_CopyToTPRegion` | identity fwd, all-reduce bwd | Implicit in `ColwiseParallel(input_layouts=Replicate())` |
| `_AllGatherFromSPRegion` | all-gather fwd (SP -> TP), reduce-scatter bwd | `PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),))` |
| `_ReduceScatterToSPRegion` | reduce-scatter fwd (TP -> SP), all-gather bwd | `RowwiseParallel(output_layouts=Shard(1))` |
| `ColumnParallelLinear` | shard weight on output dim, `_CopyToTPRegion` on input | `ColwiseParallel()` on `nn.Linear` |
| `RowParallelLinear` | shard weight on input dim, `_ReduceScatterToSPRegion` on output | `RowwiseParallel(output_layouts=Shard(1))` on `nn.Linear` |
| LayerNorm on `(B, S/N, h)` manually | element-wise on sharded sequence | `SequenceParallel()` on `nn.LayerNorm` |
| `_AllGatherFromParallelRegion` + `tp_cross_entropy` | all-gather vocab-sharded logits for loss | `loss_parallel()` context manager |
| Manual `x[:, start:start+S_local, :]` scatter | slice embedding output to SP region | `RowwiseParallel(output_layouts=Shard(1))` on `nn.Embedding` |


### Data Flow: Hand-Written vs DTensor

The following diagram traces one transformer block. The left column shows the hand-written approach; the right column shows the DTensor approach. Both execute the same collectives and produce the same shapes.

```
      HAND-WRITTEN (model_gpt_tp_sp.py)             DTensor (model_gpt_tp_sp_dtensor.py)

      x: local Tensor (B, S/N, h)                   x: DTensor Shard(1) (B, S/N, h)
                  |                                             |
            [LayerNorm]                                 [SequenceParallel() on norm1]
            operates on local (B, S/N, h)               operates on Shard(1) DTensor
                  |                                             |
     _AllGatherFromSPRegion.apply()                 PrepareModuleInput:
     all-gather: (B,S/N,h) -> (B,S,h)              Shard(1) -> Replicate [all-gather]
                  |                                             |
        [ColumnParallelLinear]                       [ColwiseParallel on W_q,W_k,W_v]
        _CopyToTPRegion (identity fwd)              implicit Replicate input
        each W: (d_model, d_model/N)                weight Shard(0), output Shard(-1)
                  |                                             |
           [Attention]                                  [F.scaled_dot_product_attention]
        manual Q @ K.T, softmax, @ V                native DTensor SDPA
                  |                                             |
        [RowParallelLinear W_o]                      [RowwiseParallel on W_o]
        _ReduceScatterToSPRegion.apply()            output_layouts=Shard(1)
        reduce-scatter: (B,S,h) -> (B,S/N,h)       [reduce-scatter]
                  |                                             |
            [Dropout]                                   [Dropout]
        local Tensor (B, S/N, h)                    DTensor Shard(1) (propagated)
                  |                                             |
         residual + dropout                            residual + dropout
        local + local = local                       DTensor + DTensor = DTensor
                  |                                             |
            [LayerNorm]                                 [SequenceParallel() on norm2]
                  |                                             |
                (... FFN sub-block, same pattern ...)
```

The collectives are identical. The difference is *who inserts them*: in the hand-written version, the autograd functions are explicit calls in the module code. In the DTensor version, `parallelize_module()` registers forward pre/post hooks that perform the same redistributions.


### The Debugging Journey

Translating the hand-written code to DTensor was not a matter of simple substitution. Two bugs exposed fundamental concepts about how DTensor works.


#### Bug 1: `output_layouts` Controls the Collective, Not `use_local_output`

The first attempt used `RowwiseParallel()` with default arguments on `W_o` and `W2`. This produced:

```
RuntimeError: The size of tensor a (128) must match the size of
tensor b (256) at non-singleton dimension 1
```

The error occurred at the residual add: `x` had shape `(B, 128, d_model)` (sequence-sharded), but the attention output had shape `(B, 256, d_model)` (full sequence).

The root cause: `RowwiseParallel()` defaults to `output_layouts=Replicate()`, which triggers an **all-reduce**. This is the vanilla TP behavior -- sum partial results and replicate. But TP+SP needs a **reduce-scatter**: sum partial results and scatter along the sequence dimension. The fix is `output_layouts=Shard(1)`.

A natural follow-up question: "Could we use `use_local_output=False` instead?" No. These two parameters control orthogonal things:

```
                     RowwiseParallel on W_o
                     partial sums on each GPU
                              |
                    output_layouts = ?
                     /                  \
              Replicate()            Shard(1)
              [ALL-REDUCE]          [REDUCE-SCATTER]
              (B, S, h)             (B, S/N, h)
                  |                      |
           use_local_output = ?    use_local_output = ?
            /           \           /           \
          True         False      True         False
       local Tensor   DTensor   local Tensor   DTensor
       (B, S, h)    Replicate   (B, S/N, h)   Shard(1)
                    (B, S, h)                 (B, S/N, h)
```

`output_layouts` determines **which collective** runs and therefore **what shape** results. `use_local_output` only determines **whether the result is unwrapped** from a DTensor to a plain `torch.Tensor`. Confusing them is the single most common DTensor mistake when implementing SP.

| `RowwiseParallel` parameter | What it controls | Hand-written equivalent |
|---|---|---|
| `output_layouts=Replicate()` (default) | all-reduce -- sum partials, replicate result | `_ReduceFromTPRegion` (vanilla TP) |
| `output_layouts=Shard(1)` | reduce-scatter -- sum partials, scatter along seq dim | `_ReduceScatterToSPRegion` (TP+SP) |
| `use_local_output=True` (default) | unwrap DTensor to plain `torch.Tensor` | `.to_local()` |
| `use_local_output=False` | keep result as DTensor with placement metadata | (no unwrap) |


#### Bug 2: Mixed Tensor/DTensor at Residual Adds

After fixing the collective, the next error appeared:

```
RuntimeError: aten.add.Tensor: got mixed torch.Tensor and DTensor,
need to convert all torch.Tensor to DTensor before calling
distributed operators!
```

This occurred at the same residual add line:

```python
x = x + self.resid_dropout(self.attn(self.norm1(x)))
```

The problem: `x` entering the block is a `Shard(1)` **DTensor** (from the embedding layer with `use_local_output=False`). But `W_o` with `use_local_output=True` (the default) unwraps its output to a plain **local tensor**. Dropout passes the local tensor through unchanged. The residual add then sees `DTensor + Tensor` -- which PyTorch rejects.

```
    emb output              norm1 output        W_o output            residual add
    +----------+          +----------+         +----------+          +----------+
    | DTensor  |  ---->   | DTensor  |  ---->  |   ???    |  ---->   | x + ???  |
    | Shard(1) |          | Shard(1) |         |          |          |          |
    +----------+          +----------+         +----------+          +----------+

    Option A: use_local_output=True on W_o (default)
              W_o output = local Tensor
              Dropout output = local Tensor
              x (DTensor) + local Tensor = ERROR

    Option B: use_local_output=False on W_o
              W_o output = DTensor Shard(1)
              Dropout output = DTensor Shard(1) (element-wise ops propagate)
              x (DTensor) + DTensor = OK
```

Two solutions exist:

**Solution A:** Add `SequenceParallel()` to `resid_dropout` in the plan. This registers a hook that wraps the incoming local tensor as a `Shard(1)` DTensor, so the residual add becomes DTensor + DTensor. This is explicit but adds a plan entry that only exists to fix a type mismatch.

**Solution B:** Set `use_local_output=False` on `W_o` and `W2`. Their outputs stay as `Shard(1)` DTensors. `nn.Dropout` is element-wise, so it propagates the DTensor placement naturally. No extra plan entry needed.

We chose Solution B. The resulting rule is simple:


### The DTensor Consistency Rule

Once any module in a computation path outputs a DTensor, **every path to every downstream operation must produce DTensors with compatible placements**. PyTorch does not auto-promote plain tensors to DTensors at binary operations like `+`.

In practice, this means: if the embeddings output `Shard(1)` DTensors (`use_local_output=False`), then the entire residual stream is DTensors. Every branch feeding into a residual add must also produce DTensors. Setting `use_local_output=False` on `W_o` and `W2` ensures this.

The alternative is to keep everything as local tensors (`use_local_output=True` everywhere) and use `SequenceParallel()` plan entries on intermediate modules like dropout to bridge the gaps. This works but adds plan entries that exist only for type management, not for expressing parallelism intent.


### Code Eliminated

The DTensor approach eliminates all parallelism plumbing from the model definition:

```
Component                                Hand-written    DTensor
-------------------------------------------------------------------
Autograd primitives (3 classes)              30 lines    0 lines
Communication helpers                        15 lines    0 lines
ColumnParallelLinear                         18 lines    0 lines
RowParallelLinear                            20 lines    0 lines
TPAttention (with manual sharding)           35 lines    0 lines
TPFFN (with manual sharding)                 10 lines    0 lines
_AllGatherFromParallelRegion + tp_cross_entropy  18 lines    0 lines
-------------------------------------------------------------------
Subtotal (parallelism plumbing)            ~146 lines    0 lines
apply_tp_sp() sharding plan                   0 lines  ~60 lines
-------------------------------------------------------------------
Net reduction                                           ~86 lines
```

More importantly, the model definition is now reusable. The same `GPT` class can run on a single GPU (no parallelism), with pure TP (different plan), or with TP+SP (the plan shown here). The parallelism strategy is a configuration concern, not an architectural one.


### Composing with FSDP: The Path to Multi-Dimensional Parallelism

One of the strongest arguments for DTensor-based TP is composability. In practice, large-scale training combines TP (intra-host, over fast NVLink) with FSDP (inter-host, over network). With the hand-written approach, adding FSDP would require wrapping our custom `ColumnParallelLinear` and `RowParallelLinear` modules with FSDP, handling mixed DTensor/plain-tensor parameters, and ensuring the communication groups do not conflict.

With DTensor, this is a 2-D `DeviceMesh` and two function calls:

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# 2-D mesh: 8-way DP across hosts, 8-way TP within each host
mesh_2d = init_device_mesh("cuda", (8, 8), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh_2d["tp"]
dp_mesh = mesh_2d["dp"]

model = GPT(...)
apply_tp_sp(model, tp_mesh)       # TP intra-host
model = fully_shard(model, mesh=dp_mesh)  # FSDP inter-host
```

The model code is unchanged. The TP plan is unchanged. Only the mesh shape and the FSDP wrapping are new. This composability is the practical payoff of decoupling the model from the parallelism strategy.


### Running and Comparing

From the `sequence-parallelism/` directory:

```bash
# Hand-written TP+SP
torchrun --nproc_per_node=2 src/model_gpt_tp_sp.py

# DTensor TP+SP
torchrun --nproc_per_node=2 src/model_gpt_tp_sp_dtensor.py

# With more GPUs (n_heads must be divisible by world size)
torchrun --nproc_per_node=4 src/model_gpt_tp_sp_dtensor.py --n-heads 8
```

Both scripts produce JSON benchmark results in `outputs/`. The loss values will differ slightly due to different RNG paths (DTensor's weight initialization uses `distribute_tensor` which may shard differently) and the use of `F.scaled_dot_product_attention` (fused kernel) vs manual `Q @ K.T` (explicit matmul). The shapes, parameter counts, and communication patterns are identical.


### Source Code

| File | Description |
|---|---|
| [model_gpt_tp_sp.py](sequence-parallelism/src/model_gpt_tp_sp.py) | Hand-written TP+SP with custom autograd primitives |
| [model_gpt_tp_sp_dtensor.py](sequence-parallelism/src/model_gpt_tp_sp_dtensor.py) | DTensor TP+SP with `parallelize_module()` sharding plan |
| [model_gpt_tp.py](sequence-parallelism/src/model_gpt_tp.py) | Vanilla TP baseline (for comparison) |
| [sequence-parallelism.md](sequence-parallelism.md) | Full TP+SP theory, primitives, and activation analysis |


### References

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," [arXiv:1909.08053](https://arxiv.org/abs/1909.08053), 2019.
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," [arXiv:2205.05198](https://arxiv.org/abs/2205.05198), 2022.
- PyTorch, "Large Scale Transformer model training with Tensor Parallel (TP)," [docs.pytorch.org/tutorials/intermediate/TP_tutorial.html](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html).
- PyTorch, "Tensor Parallel APIs," [docs.pytorch.org/docs/stable/distributed.tensor.parallel.html](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html).
