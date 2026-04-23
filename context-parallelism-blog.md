## Context Parallelism from Scratch: Ring Attention for Million-Token Sequences

*This is Part 5 of a six-part series on model parallelism. Parts [1](tensor-parallelism-blog.md) and [2](tensor-parallelism-dtensor.md) cover Tensor Parallelism. Parts [3](sequence-parallelism-blog.md) and [4](sequence-parallelism-dtensor.md) cover Sequence Parallelism. [Part 6](context-parallelism-dtensor.md) translates this hand-written implementation to PyTorch's DTensor Context Parallel API.*

Context length in large language models has grown from 2K tokens (GPT-2) to 128K (Llama 3.1) to over 1M (Gemini 1.5 Pro). But inside the attention layer, every query must touch every key it is allowed to attend to, and the intermediate score matrix scales as $O(S^2)$. Tensor Parallelism splits heads, Sequence Parallelism splits activations outside the TP region -- but neither touches the quadratic attention bottleneck. Context Parallelism (CP) does.

This article builds ring attention from scratch: the P2P rotation of K/V blocks around a ring of GPUs, the online softmax that merges partial results without ever materializing the full score matrix, and the load balancing that makes causal masking efficient. The standalone primitives will live in `ring_attention.py` and the full model integration in `model_gpt_cp.py` (both under [context-parallelism/src/](context-parallelism/src/)). The naive CP microbenchmark is in [step2_cp_comparison.py](context-parallelism/src/step2_cp_comparison.py).


### A Map Before the Territory

The terminology around long-context attention is dense: SDPA, Flash Attention, Ring Attention, Context Parallelism, zig-zag attention, online softmax, load balancer. These are not competing alternatives -- they are layers in a stack. Understanding which layer each concept belongs to is the key to clarity.

We organize them into four layers, then five implementation components, and then deep-dive into each.


### Layer 1: The Single-GPU Problem

Standard attention computes $\mathrm{softmax}(QK^\top / \sqrt{d})\, V$. The naive way materializes the full $S \times S$ score matrix in GPU memory.

**SDPA** (`F.scaled_dot_product_attention`) is PyTorch's API for attention. It is just a function signature. Under the hood, PyTorch picks a *kernel* based on your inputs and hardware.

**Flash Attention** is a specific kernel (Dao et al.) that computes attention on a single GPU without materializing the full $S \times S$ matrix. It tiles Q into blocks, streams K/V blocks from HBM to SRAM, and uses **online softmax** to merge partial results. Peak memory drops from $O(S^2)$ to $O(S)$.

**Online softmax** is the mathematical technique that makes tiling possible. You keep three running quantities -- the current maximum $m$, the sum of exponentials $\ell$, and the unnormalized output accumulator $o$ -- and rescale whenever a new block shifts the maximum. After all blocks, $o / \ell$ gives the exact softmax-weighted output.

```mermaid
flowchart LR
  subgraph layer1 ["Layer 1: Single GPU"]
    SDPA["F.scaled_dot_product_attention\n(PyTorch API)"] --> dispatch{Kernel dispatch}
    dispatch --> mathKernel["Math kernel\n(naive S x S)"]
    dispatch --> flashKernel["Flash Attention kernel\n(tiled, no S x S)"]
    dispatch --> cuDNN["cuDNN kernel"]
    flashKernel --> onlineSoftmax["Online Softmax\n(m, l, o) recurrence"]
  end
```

The takeaway: SDPA is the interface, Flash is the implementation, online softmax is the math. All on one GPU.


### Layer 2: The Multi-GPU Problem -- Context Parallelism

When S is so large that even Flash Attention on one GPU runs out of memory -- the Q/K/V tensors themselves don't fit, or you need the memory budget for other activations -- you split the sequence across GPUs.

**Context Parallelism (CP)** is the umbrella term for any strategy that shards the sequence dimension across GPUs for the attention computation. It has two main implementations:

**Naive CP:** Each GPU holds local Q ($S/C$ rows) but full K and V (all $S$ columns). The score tile per GPU is $(S/C) \times S$. Memory savings are linear in C, but you store full K/V everywhere. The [step2_cp_comparison.py](context-parallelism/src/step2_cp_comparison.py) benchmark in this repo implements this variant.

**Ring Attention:** Each GPU holds local Q ($S/C$ rows) and local K/V ($S/C$ columns). K/V blocks rotate around a ring of GPUs in $C$ steps. At each step, the score tile is only $(S/C) \times (S/C)$. Memory savings are quadratic in $C$. Ring attention uses the same online softmax from Layer 1 to merge partial results across ring steps -- the identical $(m, \ell, o)$ math, but blocks arrive from other GPUs instead of from SRAM.

```mermaid
flowchart TD
  subgraph layer2 ["Layer 2: Context Parallelism"]
    CP["Context Parallelism\n(umbrella strategy)"]
    CP --> naiveCP["Naive CP\nlocal Q, full K/V\nscore: S/C x S"]
    CP --> ringAttn["Ring Attention\nlocal Q, rotating K/V\nscore: S/C x S/C per step"]
    ringAttn --> allToAll["All-to-all rotation\n(P2P send/recv ring)"]
    ringAttn --> allGather["All-gather rotation\n(gather full K/V)"]
    ringAttn --> merger["Online Softmax Merger\nsame (m, l, o) math as Flash,\nbut blocks arrive from other GPUs"]
  end
```

Two transport variants exist for the ring:
- **All-to-all rotation (P2P ring):** Each GPU sends its current K/V to the next neighbor and receives from the previous. After $C$ steps, every GPU has seen every block. This is what we build in this article.
- **All-gather rotation:** Each GPU all-gathers the full K/V, computes, and discards. Simpler but higher peak memory. Used in Llama 3 training.


### Layer 3: Load Balancing

With causal masking, early tokens attend to few keys and late tokens attend to many. If you assign contiguous chunks to GPUs, the last rank does far more work than the first.

**No load balancing (contiguous):** rank 0 gets tokens `[0,1,2,3]`, rank 1 gets `[4,5,6,7]`. With S=8 and C=2, rank 0 computes 10 score entries, rank 1 computes 26. Rank 1 is 2.6x slower.

**Head-tail / zig-zag:** rank 0 gets `[0,7,1,6]`, rank 1 gets `[2,5,3,4]`. Each rank pairs an early token (few keys) with a late token (many keys): 18 entries each. Perfectly balanced.

```mermaid
flowchart LR
  subgraph layer3 ["Layer 3: Load Balancing"]
    causal["Causal mask\nearly tokens: light\nlate tokens: heavy"]
    causal --> contiguous["Contiguous chunks\nrank0=[0,1,2,3]\nrank1=[4,5,6,7]\nIMBALANCED"]
    causal --> headTail["Head-tail / zig-zag\nrank0=[0,7,1,6]\nrank1=[2,5,3,4]\nBALANCED"]
  end
```

Load balancing is orthogonal to the ring vs naive choice. It requires `seq_len % (2 * CP) == 0` because each rank must get an equal count of head and tail tokens. The "zig-zag attention" and "striped attention" names in the literature refer to ring attention with this load-balanced token assignment.


### Layer 4: How PyTorch Wires It All Together

In production (torchtitan, PyTorch 2.7+), the user writes plain `F.scaled_dot_product_attention`. The `_ContextParallel` plan intercepts the call, applies load balancing, runs ring attention (which internally calls Flash kernels per step), merges via online softmax, restores original token order, and returns the output. The user never sees any of this.

```mermaid
flowchart TD
  subgraph layer4 ["Layer 4: Full Pipeline in PyTorch"]
    userCode["User calls\nF.scaled_dot_product_attention"] --> interceptCP["_ContextParallel plan\nintercepts SDPA call"]
    interceptCP --> loadBalance["_HeadTailLoadBalancer\nreorders tokens"]
    loadBalance --> ringLoop["Ring Attention loop:\nfor each step:\n  recv K/V block\n  Flash kernel on local tile\n  online softmax merge\n  send K/V block"]
    ringLoop --> restore["Load balancer\nrestores original order"]
    restore --> output["Output identical to\nsingle-GPU full attention"]
  end
```

Part 6 of this series will show this Layer 4 integration in detail. This article focuses on Layers 1-3: building ring attention by hand.


### The 5 Implementation Components

The torchtitan blog identifies five components that make up a Context Parallel implementation. They map onto our four layers:

| # | Component | Layer | "Understand it" | "Implement it" |
|---|-----------|-------|-----------------|----------------|
| i | Tensor sharding | 2 | How to partition `[B, S]` inputs across the CP group | `_context_parallel_shard()` |
| ii | Attention op dispatch | 4 | Intercepting SDPA and routing to ring attention | `_ContextParallel` plan |
| iii | Shard rotation | 2 | Moving K/V blocks between GPUs (all-to-all or all-gather) | `_ring_rotate()` with `batch_isend_irecv` |
| iv | Load balancer | 3 | Reordering tokens for balanced causal work | `_HeadTailLoadBalancer` |
| v | SDPA merger | 1 | Combining partial results via online softmax | $(m, \ell, o)$ recurrence |

The layered view tells you *why* each component exists. The 5-component view tells you *what to build*. We now deep-dive into each, bottom-up.


### The Bottleneck TP and SP Leave Behind

Attention (single head, schematic):

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

Shapes (batch $B$, heads $H$, sequence $S$, head dim $d$):

- $Q, K, V$: $(B, H, S, d)$
- Scores $Q K^\top$: $(B, H, S, S)$

**Tensor parallelism** splits heads: local scores are $(B, H/N, S, S)$. Smaller in $H$ but still $O(S^2)$ in the sequence dimensions.

**Sequence parallelism** (Megatron-style) keeps the TP region at full $S$. Attention scores stay at $(B, H/N, S, S)$ inside the TP region. SP saves memory on LayerNorm, Dropout, and residuals -- not on the quadratic attention tile.

![CP high-level](context-parallelism/images/cp.png)

The memory table makes the bottleneck concrete:

| Sequence Length | Attention Memory (per layer, B=1, H=32, bf16) |
|---|---|
| 1,024 | 64 MB |
| 4,096 | 1 GB |
| 16,384 | 16 GB |
| 65,536 | 256 GB -- impossible on any single GPU |
| 131,072 | 1,024 GB |

For very large $S$, the limiting object is the attention score matrix, not the linear layers.


### Naive CP: The Simple Baseline

The simplest form of context parallelism: shard Q across GPUs, but replicate full K and V everywhere.

Each GPU computes attention for its local queries against all keys:

```
Without CP:  Each GPU computes (S x S) attention scores
With CP=C:   Each GPU computes (S/C x S) attention scores

Score tile shrinks by C on one axis only.
```

The [step2_cp_comparison.py](context-parallelism/src/step2_cp_comparison.py) benchmark measures this directly. With 2 GPUs and S=4096, memory drops roughly in half because the score matrix goes from $(S, S)$ to $(S/2, S)$.

**Why naive CP is limited:**
- Memory for scores is $O(S^2/C)$ -- linear savings only.
- Full K and V are replicated on every GPU -- no savings on K/V storage.
- No inter-GPU communication during attention, but a big all-gather or broadcast of K/V is needed up front.

Ring attention solves all three problems.


### Ring Attention: The Efficient Solution

![CP attention layer](context-parallelism/images/cp-attn.png)

Ring attention restructures the computation as two nested loops ([Coconut Mode](https://coconut-mode.com/posts/ring-attention/)):

- **Outer loop** over Q chunks: fully parallel across GPUs. Each GPU owns one $Q_i$ permanently.
- **Inner loop** over K/V chunks: computed iteratively via the ring. At each step, a new $(K_j, V_j)$ block arrives from the previous neighbor.

**Why splitting Q is easy:** Each output row depends on only one query row. Assign query chunks to GPUs and the outputs are independent.

**Why splitting K/V is hard:** Softmax normalizes over the entire key axis. You cannot compute the normalization constant without seeing all keys. Online softmax resolves this by accumulating the normalizer incrementally.

![Ring attention topology](context-parallelism/images/ring-attn.png)

```mermaid
flowchart TD
  subgraph outerLoop ["Outer loop (parallel across GPUs)"]
    GPU0["GPU 0: owns Q_0"]
    GPU1["GPU 1: owns Q_1"]
    GPU2["GPU 2: owns Q_2"]
    GPU3["GPU 3: owns Q_3"]
  end
  subgraph innerLoop ["Inner loop (ring rotation, C steps)"]
    step0["Step 0: compute Q_i @ K_local^T"]
    step1["Step 1: rotate K/V, compute Q_i @ K_next^T"]
    step2["Step 2: rotate K/V, compute Q_i @ K_next^T"]
    step3["Step 3: rotate K/V, compute Q_i @ K_next^T"]
    step0 --> step1 --> step2 --> step3
  end
  GPU0 --> innerLoop
  GPU1 --> innerLoop
  GPU2 --> innerLoop
  GPU3 --> innerLoop
  innerLoop --> merge["Online softmax merge\nafter C steps: output = o / l"]
```


#### Ring KV Rotation

At each ring step, every GPU sends its current K/V block to the next neighbor and receives from the previous. After $C$ steps, every GPU has seen every K/V block.

```
  GPU0 ----KV----> GPU1 ----KV----> GPU2 ----KV----> GPU3
    ^                                                |
    +------------------- KV -------------------------+
```

The rotation uses `torch.distributed.batch_isend_irecv` for efficient P2P:

<!-- TODO: Replace with code from context-parallelism/src/ring_attention.py once built -->

```python
def _ring_rotate(k, v, cp_group):
    """Send KV to next rank, receive from previous rank."""
    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    next_rank = (cp_rank + 1) % cp_size
    prev_rank = (cp_rank - 1) % cp_size

    global_ranks = dist.get_process_group_ranks(cp_group)
    k_new = torch.empty_like(k)
    v_new = torch.empty_like(v)

    p2p_ops = [
        dist.P2POp(dist.isend, k.contiguous(), global_ranks[next_rank]),
        dist.P2POp(dist.irecv, k_new, global_ranks[prev_rank]),
        dist.P2POp(dist.isend, v.contiguous(), global_ranks[next_rank]),
        dist.P2POp(dist.irecv, v_new, global_ranks[prev_rank]),
    ]
    reqs = dist.batch_isend_irecv(p2p_ops)
    for req in reqs:
        req.wait()

    return k_new, v_new
```

Each call performs four non-blocking operations in a batch: two sends and two receives. The `batch_isend_irecv` groups them into a single NCCL call for efficiency.


#### Communication-Computation Overlap

The key performance insight: while GPU $i$ computes `Q_i @ K_j^T`, it can simultaneously send $K_j, V_j$ to GPU $i+1$ and receive $K_{j-1}, V_{j-1}$ from GPU $i-1$. If the compute takes longer than the transfer, communication is completely hidden.

The overlap condition ([Coconut Mode](https://coconut-mode.com/posts/ring-attention/#memory-and-arithmetic-complexity)): communication time is $4 \cdot c \cdot d / B$ (transferring K and V blocks at bandwidth $B$). Compute time is approximately $4 \cdot d \cdot c^2 / F$ (two matmuls at $F$ flops/sec). Communication is fully hidden when:

$$
\frac{4 \cdot c \cdot d}{B} \le \frac{4 \cdot d \cdot c^2}{F} \quad \Longrightarrow \quad c \ge \frac{F}{B} \quad \Longrightarrow \quad \frac{S}{C} \ge \frac{F}{B}
$$

where $c = S/C$ is the chunk size per GPU. For the long sequences where you need CP (32K+ tokens), this is easily satisfied on modern hardware. The ring overhead is effectively zero.


#### Causal Masking in a Ring

With causal attention, a query at position $i$ can only attend to keys at positions $\le i$. When K/V blocks arrive from different ranks, the masking rule depends on which chunk the block came from:

| Source rank vs current rank | Masking rule |
|---|---|
| `source_rank < cp_rank` | Past chunk: no masking (attend to everything) |
| `source_rank == cp_rank` | Same chunk: standard upper-triangular causal mask |
| `source_rank > cp_rank` | Future chunk: full mask (attend to nothing, skip computation) |

In code, this is a simple conditional inside the ring loop:

<!-- TODO: Replace with code from context-parallelism/src/ring_attention.py once built -->

```python
if source_rank == cp_rank:
    causal_mask = torch.triu(
        torch.ones(T_local, T_local, device=q.device, dtype=torch.bool),
        diagonal=1
    )
    scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
elif source_rank > cp_rank:
    scores.fill_(float('-inf'))
# else: source_rank < cp_rank -> past chunk, no masking needed
```

Future-chunk blocks can be skipped entirely (no compute needed), which is an optimization opportunity.


#### Online Softmax Merger

$\mathrm{softmax}$ normalizes over the key axis. For one query row, let $s \in \mathbb{R}^T$ be scores against $T$ keys:

$$
a = \sum_{j=1}^{T} \frac{e^{s_j}}{\sum_{k=1}^{T} e^{s_k}} \, V_j
$$

In ring attention, you never store the full $s$. You see blocks $s^{(1)}, s^{(2)}, \ldots$ arriving from the ring. Keep three running quantities:

| Symbol | Meaning |
|--------|---------|
| $m$ | Running maximum of all scores seen so far |
| $\ell$ | $\sum_{j \in \text{seen}} e^{s_j - m}$ (sum of exponentials relative to current $m$) |
| $o$ | $\sum_{j \in \text{seen}} e^{s_j - m} V_j$ (unnormalized weighted sum at current $m$) |

When a new block arrives with scores $s'$ and values $V'$, update:

$$
m' = \max(m,\, \max(s'))
$$

$$
\ell_{\text{new}} = \ell \cdot e^{m - m'} + \sum_{i \in \text{block}} e^{s'_i - m'}
$$

$$
o_{\text{new}} = o \cdot e^{m - m'} + \sum_{i \in \text{block}} e^{s'_i - m'}\, V'_i
$$

The $e^{m - m'}$ factor rescales everything accumulated under the old maximum to the new reference. After the last block, $a = o / \ell$ gives the exact result.

**Why this is exact:** The $e^{-m}$ factor cancels between numerator and denominator of softmax. At the end, $m$ equals the global maximum over all keys, and the recurrence has accumulated the correct sums.

**The same math powers both Flash Attention (tiling within one GPU's SRAM) and Ring Attention (tiling across GPUs).** The only difference is where the blocks come from.


#### Putting It Together

The full ring attention forward combines rotation, causal masking, and online softmax into a single loop:

<!-- TODO: Replace with code from context-parallelism/src/ring_attention.py once built -->

```python
def ring_attention_forward(q_local, k_local, v_local, cp_group):
    """Ring attention: each GPU owns Q_local, rotates K/V around the ring."""
    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    B, T_local, n_head, head_dim = q_local.shape
    scale = head_dim ** -0.5

    q = q_local.transpose(1, 2).float()  # (B, n_head, T_local, head_dim)

    # Online softmax accumulators
    o_acc = torch.zeros(B, n_head, T_local, head_dim, device=q.device, dtype=torch.float32)
    m = torch.full((B, n_head, T_local, 1), float('-inf'), device=q.device, dtype=torch.float32)
    l = torch.zeros(B, n_head, T_local, 1, device=q.device, dtype=torch.float32)

    k_recv = k_local.clone().transpose(1, 2).float()
    v_recv = v_local.clone().transpose(1, 2).float()

    for step in range(cp_size):
        source_rank = (cp_rank - step) % cp_size

        # Attention scores for this block
        scores = torch.matmul(q, k_recv.transpose(-2, -1)) * scale

        # Causal masking
        if source_rank == cp_rank:
            causal = torch.triu(torch.ones(T_local, T_local, device=q.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        elif source_rank > cp_rank:
            scores.fill_(float('-inf'))

        # Online softmax update
        block_max = scores.max(dim=-1, keepdim=True).values.clamp(min=-1e30)
        block_exp = torch.exp(scores - block_max)
        block_sum = block_exp.sum(dim=-1, keepdim=True)
        block_out = torch.matmul(block_exp, v_recv)

        new_m = torch.maximum(m, block_max)
        exp_old = torch.exp(m - new_m)
        exp_new = torch.exp(block_max - new_m)

        l = exp_old * l + exp_new * block_sum
        o_acc = exp_old * o_acc + exp_new * block_out
        m = new_m

        # Rotate KV to next rank
        if step < cp_size - 1:
            k_send = k_recv.transpose(1, 2)
            v_send = v_recv.transpose(1, 2)
            k_rotated, v_rotated = _ring_rotate(k_send, v_send, cp_group)
            k_recv = k_rotated.transpose(1, 2).float()
            v_recv = v_rotated.transpose(1, 2).float()

    output = (o_acc / l.clamp(min=1e-8)).transpose(1, 2)
    return output.to(q_local.dtype)
```

**Walkthrough with 4 GPUs (C=4, S=16):**

| Step | GPU 0 holds K/V from | GPU 1 holds K/V from | GPU 2 holds K/V from | GPU 3 holds K/V from |
|---|---|---|---|---|
| 0 | rank 0 (local) | rank 1 (local) | rank 2 (local) | rank 3 (local) |
| 1 | rank 3 | rank 0 | rank 1 | rank 2 |
| 2 | rank 2 | rank 3 | rank 0 | rank 1 |
| 3 | rank 1 | rank 2 | rank 3 | rank 0 |

After 4 steps, every GPU has seen every K/V block. The online softmax accumulators hold the exact attention output.


### Load Balancing: Contiguous vs Head-Tail

With contiguous token assignment and causal masking, the computational imbalance is severe. For S=8, C=2:

```
Contiguous assignment:
  rank 0: tokens [0,1,2,3]  ->  1+2+3+4 = 10 score entries
  rank 1: tokens [4,5,6,7]  ->  5+6+7+8 = 26 score entries
  Rank 1 does 2.6x more work!
```

**Head-tail reordering** (also called zig-zag) pairs tokens from both ends of the sequence:

```
Head-tail assignment:
  rank 0: tokens [0,7,1,6]  ->  1+8+2+7 = 18 score entries
  rank 1: tokens [2,5,3,4]  ->  3+6+4+5 = 18 score entries
  Perfectly balanced!
```

The reordering indices for S=8, C=2 are `[0, 7, 1, 6, 2, 5, 3, 4]`. In general, for rank $r$:

```python
k = seq_len // (2 * cp_world_size)
for rank in range(cp_world_size):
    reordered[rank * 2*k : (rank+1) * 2*k] = cat(
        tokens[rank * k : (rank+1) * k],          # head chunk
        tokens[-(rank+1) * k : -rank * k or None]  # tail chunk
    )
```

This is exactly what PyTorch's `_HeadTailLoadBalancer` implements. The constraint `seq_len % (2 * CP) == 0` ensures equal pairing. In torchtitan, this shows up as the validation `seq_len % (TP * 2 * CP) == 0`.

The causal mask under head-tail reordering (from PyTorch's `_load_balancer.py`):

```
Contiguous (imbalanced):          Head-tail (balanced):
  rank 0: [1,0,0,0,0,0,0,0]        rank 0: [1,0,0,0,0,0,0,0]
          [1,1,0,0,0,0,0,0]                [1,1,1,1,1,1,1,1]
          [1,1,1,0,0,0,0,0]                [1,1,0,0,0,0,0,0]
          [1,1,1,1,0,0,0,0]                [1,1,1,1,1,1,1,0]
  ---------------------------       ---------------------------
  rank 1: [1,1,1,1,1,0,0,0]        rank 1: [1,1,1,0,0,0,0,0]
          [1,1,1,1,1,1,0,0]                [1,1,1,1,1,1,0,0]
          [1,1,1,1,1,1,1,0]                [1,1,1,1,0,0,0,0]
          [1,1,1,1,1,1,1,1]                [1,1,1,1,1,0,0,0]
```

Count the 1s: contiguous gives 10 vs 26. Head-tail gives 18 vs 18.

<!-- TODO: Add load-balanced variant to context-parallelism/src/ring_attention.py -->


### Full Model Integration

Ring attention replaces standard attention inside the GPT model's transformer blocks. The rest of the model (embeddings, LayerNorm, FFN, output projection) operates on sequence-sharded tensors at $[B, S/C, h]$.

<!-- TODO: Build context-parallelism/src/model_gpt_cp.py -->

The integration follows the same pattern as the Vizuara workshop:

```python
def apply_context_parallelism(model, cp_group):
    """Enable ring attention in all attention modules."""
    for block in model.transformer.h:
        block.attn.cp_group = cp_group
        block.attn._ring_attn_fn = ring_attention_forward
```

Inside the attention module's forward pass, when `cp_group` is set, it calls `ring_attention_forward` instead of `F.scaled_dot_product_attention`.

```bash
# Run with CP=2 on 8 GPUs (4 data-parallel replicas x 2 CP)
torchrun --nproc_per_node=8 context-parallelism/src/model_gpt_cp.py --cp_size 2

# Run with CP=4
torchrun --nproc_per_node=8 context-parallelism/src/model_gpt_cp.py --cp_size 4
```

<!-- TODO: Build context-parallelism/src/test_ring_attention.py for correctness verification -->


### Activation Shape Trace

For a transformer block with CP degree $C$, batch $B$, hidden dim $h$, $H$ heads, head dim $d = h/H$:

| Location | Shape | Notes |
|----------|-------|-------|
| Block input | $(B, S/C, h)$ | Sequence-sharded |
| LayerNorm output | $(B, S/C, h)$ | Local operation |
| Q, K, V projections | $(B, S/C, H, d)$ | Local linear layers |
| **Ring step score tile** | $(B, H, S/C, S/C)$ | **Per step** -- not $(S/C, S)$ |
| Online softmax state | $m, \ell$: $(B, H, S/C, 1)$; $o$: $(B, H, S/C, d)$ | Running accumulators |
| Attention output | $(B, S/C, h)$ | After all $C$ ring steps |
| FFN | $(B, S/C, h)$ | Local computation |
| Residual | $(B, S/C, h)$ | Sequence stays sharded |

The residual stream stays at $(B, S/C, h)$ throughout. No all-gather is needed between layers -- each layer's ring attention independently processes the sharded sequence.


### Communication Cost and Memory Savings

**Memory scaling:**

| Strategy | Peak score tile | Scaling |
|---|---|---|
| No CP | $(S \times S)$ per GPU | $O(S^2)$ |
| Naive CP (C GPUs) | $(S/C \times S)$ per GPU | $O(S^2/C)$ |
| Ring CP (C GPUs) | $(S/C \times S/C)$ per step | $O(S^2/C^2)$ |

**Communication per layer (ring):**
- $C$ ring steps, each transferring 2 tensors (K and V) of size $(B \times S/C \times d)$
- Total bytes per layer: $2 \times C \times B \times (S/C) \times d \times 2 = 4BSD$ bytes (independent of $C$!)
- The total data moved equals the full K/V once -- CP doesn't increase aggregate bandwidth, it just streams it

**The overlap condition:** Communication is fully hidden when $S/C \ge F/B$ (chunk size exceeds the flops-to-bandwidth ratio). On H100s with NVLink ($B \approx 450$ GB/s, $F \approx 990$ TFLOP/s), $F/B \approx 2200$ elements. For S=32K and C=8, the chunk is 4096 -- well above the threshold.

**Per-device memory:** $6 \times d \times c$ floats -- Q local + K current + V current + K receive buffer + V receive buffer + output accumulator.

**torchtitan benchmarks (PR [#592](https://github.com/pytorch/torchtitan/pull/592)):** Llama 3 8B on H100s, FSDP=8:

| CP degree | GPUs | Max seq_len | WPS/device |
|---|---|---|---|
| 1 | 8 | 32K | baseline |
| 2 | 16 | 80K | ~0.6x |
| 4 | 32 | 144K | ~0.35x |
| 8 | 64 | 300K | ~0.2x |

Max sequence length scales linearly with CP degree. MFU stays roughly constant -- the WPS drop is expected because attention flops scale quadratically with sequence length.


### What's Next

All the code in this article -- the P2P ring rotation, the online softmax merger, the causal masking logic, the load balancer -- totals roughly 115 lines of Python. In [Part 6](context-parallelism-dtensor.md), we replace all of it with a single `_ContextParallel(seq_dim=1)` plan applied via `parallelize_module`. The model stays a plain `nn.Module` with standard `F.scaled_dot_product_attention`, and PyTorch handles the ring, the merging, and the load balancing transparently.


### References

- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (Liu et al., 2023)
- [Coconut Mode: Ring Attention Explained](https://coconut-mode.com/posts/ring-attention/) -- pedagogical walkthrough of two-loop structure and overlap condition
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [torchtitan PR #592: enable Context Parallel](https://github.com/pytorch/torchtitan/pull/592) -- benchmarks and implementation
- [PyTorch Context Parallel Tutorial](https://github.com/pytorch/tutorials/blob/main/unstable_source/context_parallel.rst)
- [Striped Attention](https://arxiv.org/abs/2311.09431) (Brandon et al., 2023) -- load-balanced ring attention for causal models
- Technical notes: [context-parallelism.md](context-parallelism.md), [context_parallelism_concrete_walkthrough.md](context-parallelism/context_parallelism_concrete_walkthrough.md)
