## Tensor Parallelism from Scratch: Splitting Weight Matrices Across GPUs

*This is Part 1 of a six-part series on model parallelism. [Part 2](tensor-parallelism-dtensor.md) translates this hand-written implementation to PyTorch's DTensor API. Parts [3](sequence-parallelism-blog.md) and [4](sequence-parallelism-dtensor.md) extend the approach with Sequence Parallelism. Parts [5](context-parallelism-blog.md) and [6](context-parallelism-dtensor.md) tackle the quadratic attention bottleneck with Context Parallelism.*

When a model is too large for a single GPU, we split its weight matrices across multiple GPUs. This is Tensor Parallelism (TP), originally proposed in the [Megatron-LM](https://arxiv.org/abs/1909.08053) paper. But splitting matrices introduces a coordination problem -- GPUs need to communicate to reassemble the correct result. The primitives that manage this communication are subtle, and the payoff for understanding them is a beautiful optimization: when we chain two layers together, an entire communication round disappears.

This article builds TP from scratch using concrete matrix examples, custom `torch.autograd.Function` primitives, and a full GPT-style transformer. The complete source code is in [model_gpt_tp.py](tensor-parallelism/src/model_gpt_tp.py). The single-GPU baseline is in [model_gpt.py](tensor-parallelism/src/model_gpt.py).


### Why Tensor Parallelism?

Model sizes have grown faster than single-GPU memory. GPT-3 has 175B parameters (350 GB in FP16). Llama 2 70B has 140 GB of weights. A single A100 has 80 GB. Even at 8B parameters, the optimizer states and activations during training push memory well beyond what one GPU can hold.

Data Parallelism (DDP) replicates the entire model on every GPU. It scales training throughput by processing more data in parallel, but it does not reduce per-GPU memory -- every GPU still holds a complete copy of the model. Fully Sharded Data Parallel (FSDP) shards optimizer states and gradients, but at very large scale (hundreds of GPUs), its all-gather collectives are dominated by ring latency.

Tensor Parallelism takes a different approach: split individual weight matrices across GPUs so that each GPU stores and computes only a fraction of each layer. For an `N`-way TP split, each GPU holds roughly `1/N` of the model parameters. The cost is communication: GPUs must exchange intermediate results to reassemble the correct output.

The key question is: how much communication, and can we minimize it?


### The Math: Two Ways to Split a Matrix Multiply

Tensor Parallelism leverages two fundamental properties of matrix multiplication:

**1. Column split:** multiply the full input by each column shard independently.

```
A * B = A * [B_1 | B_2 | ...] = [A*B_1 | A*B_2 | ...]
```

Each GPU computes `A * B_i` -- a slice of the output. The results are concatenated.

**2. Row split:** multiply matching slices of input and weight, then sum.

```
A * B = [A_1 | A_2 | ...] * [B_1]   = sum(A_i * B_i)
                              [B_2]
                              [...]
```

Each GPU computes `A_i * B_i` -- a partial sum. The results must be added together.

We walk through both approaches using concrete matrices on 2 GPUs.


### Setup: The Matrices

![Setup](tensor-parallelism/images/tp-inputs.png)

Every example uses the same input matrix X and two weight matrices W1 and W2:

```
X = [0 1]       W1 = [1 3]       W2 = [5 7]
    [2 3]            [2 4]            [6 8]
    [4 5]
    [6 7]
  (4x2)            (2x2)            (2x2)
```

On a single GPU:

```
Y1 = X * W1 = [ 2   4]       Y = Y1 * W2 = [ 34   46]
              [ 8  18]                       [148  200]
              [14  32]                       [262  354]
              [20  46]                       [376  508]
```

**Goal:** get the same results when the weights are split across 2 GPUs.


### Column-Parallel Linear

**Key idea:** Split W by *columns*. Each GPU holds a vertical slice. Every GPU receives the full input X, multiplies by its local shard, and produces a *slice* of the output.

![Column-Parallel Linear](tensor-parallelism/images/tp-column.png)

W1 is (2,2). We split it by columns into two (2,1) shards:

```
GPU 0: W1_0 = [1]       GPU 1: W1_1 = [3]
              [2]                      [4]
```

Each GPU computes its local output:

```
GPU 0: X * W1_0 = [ 2]       GPU 1: X * W1_1 = [ 4]
                  [ 8]                          [18]
                  [14]                          [32]
                  [20]                          [46]
```

Each GPU holds one column of Y1. To reconstruct the full (4,2) result, we **all-gather** the outputs:

```
Y1_full = [ 2   4]  = X * W1     (correct)
          [ 8  18]
          [14  32]
          [20  46]
```

In code, column-parallel linear stores only `d_out // ws` rows of the weight (recall `nn.Linear` stores the transpose):

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        ws = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(d_out // ws, d_in))
        # ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _CopyToParallelRegion.apply(x)
        return F.linear(x, self.weight, self.bias)
```

No communication in the forward pass of the matmul itself -- only the `_CopyToParallelRegion` primitive, which is an identity in forward (the real work happens in backward, as we will see).

![Column-Parallel Primitives](tensor-parallelism/images/tp-col-prim.png)


### Row-Parallel Linear

**Key idea:** Split W by *rows*. Each GPU holds a horizontal slice. The *input* must also be split -- each GPU gets the columns of X that align with its rows of W. The outputs are *partial sums* that must be added together.

![Row-Parallel Linear](tensor-parallelism/images/tp-row.png)

W2 is (2,2). We split it by rows into two (1,2) shards:

```
GPU 0: W2_0 = [5  7]       GPU 1: W2_1 = [6  8]
```

X must also be split -- each GPU gets one column:

```
GPU 0: X_0 = [0]       GPU 1: X_1 = [1]
             [2]                     [3]
             [4]                     [5]
             [6]                     [7]
```

Each GPU computes a *partial result*:

```
GPU 0: X_0 * W2_0 = [ 0   0]       GPU 1: X_1 * W2_1 = [ 6   8]
                     [10  14]                            [18  24]
                     [20  28]                            [30  40]
                     [30  42]                            [42  56]
```

These are partial sums with the same shape. To get the full Y, we **all-reduce** (sum across GPUs):

```
Y_full = [ 0+6    0+8 ] = [ 6   8]  = X * W2     (correct)
         [10+18  14+24]   [28  38]
         [20+30  28+40]   [50  68]
         [30+42  42+56]   [72  98]
```

In code, row-parallel linear stores `d_in // ws` columns of the weight and applies all-reduce after the matmul:

```python
class RowParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        ws = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(d_out, d_in // ws))
        # Bias only on rank 0: all-reduce sums partials,
        # so if every rank added bias it would be summed ws times.
        if bias and dist.get_rank() == 0:
            self.bias = nn.Parameter(torch.empty(d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        return _ReduceFromParallelRegion.apply(out)
```

![Row-Parallel Primitives](tensor-parallelism/images/tp-row-prim.png)


### The Autograd Primitives

TP communication is implemented as `torch.autograd.Function` subclasses. Each one defines what happens in forward and backward, so gradients flow correctly through the distributed computation.

The two core primitives form a conjugate pair -- `f` and `f*` in the Megatron-LM paper:

```python
class _CopyToParallelRegion(torch.autograd.Function):
    """f: identity forward, all-reduce backward."""
    @staticmethod
    def forward(ctx, x):
        return x  # each GPU gets the same input

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad  # sum gradients from all column shards


class _ReduceFromParallelRegion(torch.autograd.Function):
    """f*: all-reduce forward, identity backward."""
    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x  # sum partial results across GPUs

    @staticmethod
    def backward(ctx, grad):
        return grad  # gradients flow unchanged
```

Why this pairing works: in the forward pass, `_CopyToParallelRegion` passes input unchanged to each GPU's column shard (identity). In the backward pass, each GPU computes gradients for its local shard, and these must be summed (all-reduce) to get the correct gradient for the replicated input.

Conversely, `_ReduceFromParallelRegion` sums partial results in forward (all-reduce), and in backward, each GPU only needs its own partial gradient (identity) because the summation distributes linearly over the backward pass.

Two additional primitives handle explicit splitting and gathering:

```python
class _ScatterToParallelRegion(torch.autograd.Function):
    """Scatter (split) in forward, all-gather in backward."""
    @staticmethod
    def forward(ctx, x):
        chunks = x.chunk(dist.get_world_size(), dim=-1)
        return chunks[dist.get_rank()].contiguous()

    @staticmethod
    def backward(ctx, grad):
        gathered = [torch.zeros_like(grad) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, grad.contiguous())
        return torch.cat(gathered, dim=-1)


class _AllGatherFromParallelRegion(torch.autograd.Function):
    """All-gather in forward, scatter in backward."""
    @staticmethod
    def forward(ctx, y):
        gathered = [torch.zeros_like(y) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, y.contiguous())
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        chunks = grad.chunk(dist.get_world_size(), dim=-1)
        return chunks[dist.get_rank()].contiguous()
```

`_ScatterToParallelRegion` and `_AllGatherFromParallelRegion` are conjugates of each other: scatter undoes gather, and vice versa. These are used for explicit input splitting (before row-parallel) and output reconstruction (after column-parallel) when the layers are used standalone. But as we will see next, when column-parallel and row-parallel are chained, these operations cancel out.


### The Cancellation: Column + Row Combined

This is the key insight that makes TP practical.

![Column + Row Combined](tensor-parallelism/images/tp-col-row.png)

In an MLP block, W1 is column-parallel and W2 is row-parallel. When used standalone:

- After column-parallel W1: each GPU has a slice of the output. An **all-gather** would reconstruct the full output.
- Before row-parallel W2: each GPU needs a slice of the input. A **scatter** would split the full input.

But look at what happens when they are chained: the column-parallel output is already split along the last dimension -- exactly the form that row-parallel input needs. The all-gather and scatter sit back-to-back and are conjugate operations. They cancel.

```python
def test_column_then_row(self):
    Y_expected = (self.X @ self.W1) @ self.W2

    X_local = _CopyToParallelRegion.apply(self.X)

    W1_local = self.W1.chunk(self.ws, dim=1)[self.rank]
    Y_col_local = X_local @ W1_local

    # NO all-gather or scatter here -- they cancel out.
    # Y_col_local is already split on dim=-1, which is what row-parallel needs.

    W2_local = self.W2.chunk(self.ws, dim=0)[self.rank]
    Y_row_local = Y_col_local @ W2_local

    Y_full = _ReduceFromParallelRegion.apply(Y_row_local)

    assert torch.allclose(Y_full, Y_expected, atol=1e-4)
```

The final all-reduce produces the correct two-layer result:

```
GPU 0: Y_col_0 @ W2_0 = [10  14]       GPU 1: Y_col_1 @ W2_1 = [ 24   32]
                         [40  56]                                [108  144]
                         [70  98]                                [192  256]
                         [100 140]                               [276  368]

All-reduce (sum): [ 34   46] = X * W1 * W2     (correct)
                  [148  200]
                  [262  354]
                  [376  508]
```

**One all-reduce per forward pass. One all-reduce per backward pass.** That is all the communication a two-layer MLP needs.

![Column + Row Primitives](tensor-parallelism/images/tp-col-row-prim.png)


### Tensor Parallelism in a Transformer Block

In a transformer block, the column-then-row pattern appears **twice**:

1. **Attention:** Q/K/V projections are column-parallel (split heads across GPUs). The output projection W_o is row-parallel. The intermediate all-gather/scatter cancels. One all-reduce per attention block.

2. **FFN/MLP:** W1 (up-projection) is column-parallel. W2 (down-projection) is row-parallel. The intermediate all-gather/scatter cancels. One all-reduce per MLP block.

Total communication per transformer layer: **2 all-reduces in forward, 2 all-reduces in backward.**

```
  x ---> [LayerNorm] ---> [Copy (f)] ---> [Q,K,V col-parallel] ---> [Attention] ---> [W_o row-parallel] ---> [Reduce (f*)] ---> + residual
                           identity                                                    all-reduce
                                                                                                                                    |
                                                                                                                                    v
          [LayerNorm] ---> [Copy (f)] ---> [W1 col-parallel] ---> [GeLU] ---> [W2 row-parallel] ---> [Reduce (f*)] ---> + residual ---> out
                           identity                                            all-reduce
```

For multi-head attention, column parallelism has a natural interpretation: each GPU computes attention for a subset of heads. The attention module stores `n_heads_local = n_heads // world_size` and reshapes accordingly:

```python
class TPAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=False):
        super().__init__()
        ws = dist.get_world_size()
        self.d_head = d_model // n_heads
        self.n_heads_local = n_heads // ws

        self.W_q = ColumnParallelLinear(d_model, d_model, bias=bias)
        self.W_k = ColumnParallelLinear(d_model, d_model, bias=bias)
        self.W_v = ColumnParallelLinear(d_model, d_model, bias=bias)
        self.W_o = RowParallelLinear(d_model, d_model, bias=bias)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads_local, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads_local, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads_local, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)  # all-reduce inside
```

Note `n_heads_local` instead of `n_heads` in the view. Each GPU only computes attention for its local subset of heads. The column-parallel W_q/W_k/W_v each output `d_model // ws` dimensions, which maps to `n_heads_local * d_head` elements. The row-parallel W_o takes this local output and all-reduces the partial sums to produce the full `d_model`-dimensional result.

The FFN is even simpler:

```python
class TPFFN(nn.Module):
    def __init__(self, d_model, d_ff, bias=True):
        super().__init__()
        self.W1 = ColumnParallelLinear(d_model, d_ff, bias=bias)
        self.W2 = RowParallelLinear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.W2(F.gelu(self.W1(x)))
```

W1 splits the `d_ff` dimension across GPUs; W2 takes the sharded intermediate and all-reduces back to `d_model`. The GeLU activation runs locally on each GPU's shard -- no communication needed for element-wise operations.


### Cross-Entropy with Sharded Logits

The LM head is column-parallel, producing vocab-sharded logits: each GPU holds `(B, S, vocab_size // N)`. Computing cross-entropy requires the full vocabulary dimension, so we all-gather before the loss:

```python
class _AllGatherFromParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ws = dist.get_world_size()
        ctx.ws = ws
        gathered = [torch.zeros_like(x) for _ in range(ws)]
        dist.all_gather(gathered, x.contiguous())
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        rank = dist.get_rank()
        chunks = grad.chunk(ctx.ws, dim=-1)
        return chunks[rank].contiguous()

def tp_cross_entropy(logits_local, labels, vocab_size):
    full = _AllGatherFromParallelRegion.apply(logits_local)
    return F.cross_entropy(full.view(-1, vocab_size), labels.view(-1))
```

This materializes the full `(batch*seq, vocab)` logits on every GPU. For large vocabularies this is wasteful -- [Part 2](tensor-parallelism-dtensor.md) shows how DTensor's `loss_parallel()` avoids this entirely.


### Attention Variants: MHA vs GQA vs MQA

The TP pattern described above assumes Multi-Head Attention (MHA), where Q, K, and V all have the same number of heads. Modern architectures use Grouped Query Attention (GQA) and Multi-Query Attention (MQA), which change the TP constraints:

| | MHA (GPT) | GQA (Llama) | MQA |
|---|---|---|---|
| Q heads | `n_heads` | `n_heads` | `n_heads` |
| KV heads | `n_heads` | `n_kv_heads` | 1 |
| W_q shape | `[d, d]` | `[d, n_heads * d_head]` | `[d, n_heads * d_head]` |
| W_k, W_v shape | `[d, d]` | `[d, n_kv_heads * d_head]` | `[d, d_head]` |
| TP constraint | `n_heads % ws == 0` | `n_heads % ws == 0` AND `n_kv_heads % ws == 0` | `ws == 1` (or replicate KV) |

In **MHA**, all projections are symmetric -- every head is sharded identically across GPUs.

In **GQA**, K/V projections are smaller. Each GPU gets a proportional subset of both Q and KV heads, and locally expands KV heads to match Q heads via `repeat_interleave` (no communication needed):

```python
# GQA: K/V projections are smaller
self.W_q = ColumnParallelLinear(d_model, n_heads * d_head)
self.W_k = ColumnParallelLinear(d_model, n_kv_heads * d_head)
self.W_v = ColumnParallelLinear(d_model, n_kv_heads * d_head)

K = self.W_k(x).view(B, T, self.n_kv_heads_local, self.d_head)
K = K.repeat_interleave(self.group_size, dim=1)  # local expansion, no comm
```

**MQA** is the extreme case with `n_kv_heads = 1`. Since we cannot split 1 head across multiple GPUs, real implementations replicate the single KV head on every rank. GQA with `n_kv_heads >= ws` was introduced precisely as the TP-friendly generalization of MQA.

See [model_llama_tp.py](tensor-parallelism/src/model_llama_tp.py) for the full GQA + TP implementation.


### Communication Summary

| Scenario | Forward comm. | Backward comm. | Comm. rounds (fwd) |
|---|---|---|---|
| Column Linear alone | 1 all-gather | 1 all-reduce + 1 scatter | 1 |
| Row Linear alone | 1 scatter + 1 all-reduce | 1 all-gather | 2 |
| Column + Row combined | 1 all-reduce | 1 all-reduce | 1 |

The combined case is what matters in practice. Each transformer layer has two column+row pairs (attention + FFN), so the total per layer is 2 all-reduces forward, 2 all-reduces backward.

In practice, TP communication overhead becomes noticeable beyond 8 GPUs. Within a single node, fast NVLink interconnects keep overhead low. Going across nodes requires slower network connections and throughput drops significantly. This is why TP is typically applied **intra-host** (within a single machine) and combined with data parallelism (FSDP) **inter-host** (across machines).


### Running the Code

From the `tensor-parallelism/` directory:

```bash
# Single-GPU baseline (no parallelism)
python src/model_gpt.py

# Hand-written Tensor Parallelism on 2 GPUs
torchrun --nproc_per_node=2 src/model_gpt_tp.py

# With more GPUs (n_heads must be divisible by world size)
torchrun --nproc_per_node=4 src/model_gpt_tp.py --n-heads 8

# Equivalence tests (verify hand primitives produce correct results)
torchrun --nproc_per_node=2 src/test_tp_primitives.py
```

Both scripts produce JSON benchmark results in `outputs/`.


### Source Code

| File | Description |
|---|---|
| [model_gpt.py](tensor-parallelism/src/model_gpt.py) | Single-GPU baseline GPT |
| [model_gpt_tp.py](tensor-parallelism/src/model_gpt_tp.py) | Hand-written TP GPT (this article) |
| [model_llama_tp.py](tensor-parallelism/src/model_llama_tp.py) | Llama with GQA + TP |
| [test_tp_primitives.py](tensor-parallelism/src/test_tp_primitives.py) | Equivalence tests for hand-written primitives |


### What's Next

Building TP from scratch reveals the essential communication pattern: column-parallel and row-parallel layers cancel each other's scatter/gather, leaving just one all-reduce per layer pair. But the implementation has a significant drawback: the model architecture is entangled with the parallelism strategy. `ColumnParallelLinear`, `RowParallelLinear`, and `TPAttention` fuse model logic with distribution code. The model cannot run on a single GPU without modification.

In [Part 2: From Hand-Written TP to PyTorch DTensor](tensor-parallelism-dtensor.md), we translate this implementation to PyTorch's `torch.distributed.tensor.parallel` API. A plain `nn.Module` with no parallelism in its definition is parallelized entirely by a declarative sharding plan -- and the same model code runs on 1 GPU or 8 without changes.

Beyond TP, there is a further optimization. In vanilla TP, LayerNorm, Dropout, and residual connections still operate on the full `(B, S, h)` tensor -- identically replicated on every GPU. [Part 3: Sequence Parallelism from Scratch](sequence-parallelism-blog.md) eliminates this redundancy by sharding along the sequence dimension between the TP regions, saving 30-50% of activation memory with zero extra communication.


### References

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," [arXiv:1909.08053](https://arxiv.org/abs/1909.08053), 2019.
- PyTorch, "Large Scale Transformer model training with Tensor Parallel (TP)," [docs.pytorch.org/tutorials/intermediate/TP_tutorial.html](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html).
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," [arXiv:2305.13245](https://arxiv.org/abs/2305.13245), 2023.
