## Context Parallelism

In [Tensor Parallelism](tensor-parallelism.md) we shard weight matrices across GPUs; each rank still participates in attention over the **full** sequence length $S$ inside the TP region. In [Sequence Parallelism](sequence-parallelism.md) we shard the sequence for LayerNorm, Dropout, and residuals, but we **all-gather** back to $(B, S, h)$ before attention and FFN. So inside attention, every GPU still materializes an attention **score** tensor of shape $(B, H, S, S)$ in the naive formulation. That piece scales as **$O(S^2)$** per layer and becomes the dominant memory bottleneck for long context.

**Context Parallelism (CP)** shards the **sequence** across a dedicated group of GPUs for the **attention** computation (and, in full training stacks, typically carries that split through the rest of the layer or model in a compatible layout). Each rank owns a chunk of queries (and, with ring-style algorithms, streams chunks of keys and values) so that no single device must hold the full $S \times S$ attention map at once.

This note ties together the same story as the Ultra-Scale Playbook's framing of memory, compute, and communication ([ultra_blog.md](ultra_blog.md)): CP trades **extra communication** (moving $K/V$ blocks between ranks) for **lower peak activation memory** on attention, enabling training and inference at tens or hundreds of thousands of tokens.


### The Bottleneck TP and SP Leave Behind

Attention (single head, schematic):

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

Shapes (batch $B$, heads $H$, sequence $S$, head dim $d$):

- $Q, K, V$: $(B, H, S, d)$
- Scores $Q K^\top$: $(B, H, S, S)$

**Tensor parallelism** splits heads: local scores are $(B, H/N, S, S)$. That is smaller in $H$ but still **$O(S^2)$** in the sequence dimensions.

**Sequence parallelism (Megatron-style)** keeps the TP region at full $S$. The activation table in the SP write-up shows that **attention scores** stay at $(B, H/N, S, S)$ in the TP region ([sequence-parallelism.md](sequence-parallelism.md) memory table). SP saves memory on LayerNorm, Dropout, and residuals, not on the quadratic attention tile.

So for very large $S$, the limiting object is often the **attention score matrix** (or equivalent materialized intermediates), not the linear layers alone.

**Figure (ASCII).** One head: every query row dotted with every key column (naive materialization). `x` marks a score we might store; causal masks zero the upper triangle in decoder models. Planned export: `cp-attention-score-sxs.png` ([Figures](#figures-planned-drawio-exports)).

```
              keys j = 0 ... S-1
            +-------------------+
queries i   | x                 |  row 0
            | x x               |  row 1
            | x x x             |  ...
            | x x x x ... x x x |  row S-1
            +-------------------+
                    S x S
```


### What Context Parallelism Changes

**Idea:** Partition tokens across $C$ GPUs (context-parallel degree $C$). Rank $r$ owns a contiguous (or otherwise assigned) chunk of length $S_r \approx S/C$ for its **queries**.

- **Ring attention** (and variants): ranks pass $K/V$ blocks around a ring in $C$ steps. At each step, a rank multiplies its local $Q$ block with the current $K/V$ block and **merges** partial results using an **online softmax** (same numerical idea as block-wise / Flash-style attention, but with blocks arriving from other GPUs).

**Compared to a naive approach:** An all-gather of full $K$ and $V$ on every rank would replicate full-sequence $K/V$ everywhere and use huge memory and bandwidth. Ring-style CP keeps **per-step** working sets smaller and overlaps communication with compute in mature implementations.

**Educational simplification:** The benchmark in this repo ([context-parallelism/src/step2_cp_comparison.py](context-parallelism/src/step2_cp_comparison.py)) implements **naive CP** below: full $K$ and $V$ on each GPU and a **local** $Q$ chunk. See the next subsection for how that differs from **ring attention**.


### Naive CP vs ring attention (peak score tile)

The same dense attention formula can be scheduled two ways. They differ in **what score matrix you materialize at one time** (order of growth ignores $B$ and $H$ factors).

| | **Naive CP** (e.g. in-class / [step2](context-parallelism/src/step2_cp_comparison.py)) | **Ring attention** |
|---|---------------------------|-------------------|
| **Queries per GPU** | $S/C$ local positions | $S/C$ local positions |
| **Keys (and values) at one matmul** | Often the **full** length-$S$ sequence on that rank | Only a **chunk** of length $S/C$ (the block in flight) |
| **Score tensor instantiated now** | **$(S/C) \times S$** | **$(S/C) \times (S/C)$** per ring step |
| **Scaling of that tensor with $S$ and $C$** | **$O(S^2 / C)$** | **$O(S^2 / C^2)$** peak per step |
| **How the wide $(S/C) \times S$ view appears** | **Explicit** in memory as one slab | **Implicit**: built over $C$ steps via **online softmax** (and matching accumulation on $V$); you never store the full wide slab as one array |

**Figure (ASCII).** Same notation: `#` is the score tile materialized **now** (one GPU, one head). Row count = local queries; column count = keys in this matmul. Planned export: `cp-naive-vs-ring-tiles.png` ([Figures](#figures-planned-drawio-exports)).

*Naive CP:* local $Q$ has $S/C$ rows; $K$ is **full** length $S$. One matmul builds the whole wide rectangle.

```
Naive CP  (peak scores on this GPU)

         K  (S columns, all keys resident)
         +------------------------+
Q local  |########################|  } S/C rows
(S/C)    +------------------------+
              (S/C) x S
```

*Ring attention:* one ring step. Local $Q$ still $S/C$ rows; only the **current** $K$ block has $S/C$ columns.

```
Ring, one step

         K_block (S/C columns)
         +--------------+
Q local  |##############|  } S/C rows
(S/C)    +--------------+
              (S/C) x (S/C)
```

After $C$ steps (and online softmax), the rank has combined its queries with **all** key blocks; the **$(S/C) \times S$** logical result never exists as one dense matrix.

**Takeaway:** Naive CP already removes the **$S \times S$** map per GPU and replaces it with **$(S/C) \times S$** -- linear savings in $C$ on **one** axis. Ring attention goes further: each step only forms a **$(S/C) \times (S/C)$** block, so peak score memory drops **quadratically** in $C$ for that tile. The lecture notes state the same contrast: naive CP is **$O((S/\mathrm{CP}) \times S)$** memory for the scores; ring attention is **$O(S^2/\mathrm{CP}^2)$** at each step because both query and key sides are local chunks.

**Same math:** After all ring steps, the output matches attending local queries to **all** keys. The difference is **workspace**, not the definition of attention.


### Logical requirement vs what CP optimizes

This distinction is easy to misunderstand, so it is worth stating in two layers: **what the math forces**, and **what engineering changes**.

**1. What dense causal attention forces (unchanged by CP).**  
For standard **dense** attention with a **full causal** mask, a query at position $i$ may attend to **every** key and value from positions $0$ through $i$. So the correct output for that query depends on **all** of those $K/V$ vectors. No amount of GPU parallelism removes that **logical** dependence unless you change the **model** (for example sliding-window attention, sparsity, or other patterns that zero out most positions). CP is **not** a trick to "skip" far-away keys while claiming full attention.

**2. What CP changes (implementation, not the formula).**  
CP changes **how and when** each device touches those keys and values:

- **Ring (or blockwise) CP:** Each rank still **must process** every $K/V$ block that its queries need, but it does so **over time** -- one (or a few) blocks resident at a step. The **full** $K$ and $V$ still **flow through** the collective work of the ranks (each block is sent and consumed); aggregate traffic is still on the order of **all** keys and values being **used** somewhere. What drops is **peak memory** on a single device (no need to hold the entire $K$ and $V$ tensors at once, and no single giant $(S,S)$ score matrix), and implementations can **overlap** communication with matmuls.

- **This repo's step2 (naive CP):** Every rank stores **full** $K$ and $V$; only $Q$ is sharded. The **attention score tile** per rank is **$(S/C) \times S$** ([naive vs ring](#naive-cp-vs-ring-attention-peak-score-tile)). That makes the "all $K/V$ still present on each GPU" case obvious. Ring-style training avoids holding full-length $K/V$ at once and uses **$(S/C) \times (S/C)$** score blocks per step while still **visiting** every $K/V$ chunk over the ring.

**3. Short summary.**  
- **Still true:** The cluster must **incorporate all $K/V$** needed for full dense attention (each block may be owned by one rank but **seen** by others when needed).  
- **What improves:** **Peak** activation memory and the **layout** of work (smaller tiles, streaming), not a free pass on **logical** all-keys attention.  
- **Do not confuse:** "Shard ownership of $K/V$ across ranks" with "avoid ever shipping or using all $K/V$"; the latter only holds if the **attention pattern** itself is sparse or local.


### Sequence Parallelism vs Context Parallelism

| | Sequence Parallelism (with TP) | Context Parallelism |
|---|-------------------------------|---------------------|
| **Where sequence is split** | Outside TP: LayerNorm, Dropout, residuals at $(B, S/C, h)$ | Attention (and often the full model) along context |
| **Sequence inside attention** | Full $S$ after all-gather into TP region | Sharded; $Q$ (and $K/V$ blocks) distributed across CP group |
| **Attention scores (peak tile)** | $(B, H/N, S, S)$ per GPU in naive TP layout | **Naive CP:** $(B, H/N, S/C, S)$. **Ring CP:** $(B, H/N, S/C, S/C)$ per step; wide row built via online softmax |
| **Main goal** | Cut memory on non-attention activations without extra comm volume vs vanilla TP | Cut **$O(S^2)$** attention working set (and enable very long $S$) |
| **Typical comm** | All-gather / reduce-scatter around TP blocks | Point-to-point or collectives moving $K/V$ (or blocks) along a ring |
| **Full $K/V$ for dense attention** | Full sequence present on each rank inside the TP attention forward (after gather) | Each rank still **consumes** every $K/V$ block over the ring (or holds full $K/V$ in simplified demos); CP cuts **peak** residency, not the logical need for all keys |

SP and CP address **different** tensors. They can be combined in large training stacks (names and exact layout depend on the framework).

**Figure (ASCII).** Where the **full** sequence lives during attention (schematic; framework details vary). Planned export: `cp-megatron-sp-vs-cp-flow.png` ([Figures](#figures-planned-drawio-exports)).

```
Megatron TP + SP (attention sub-block)

  SP region          all-gather            TP attention          reduce-scatter
  (S/C per GPU)  ->   full S on rank   ->  scores ~ S x S     -> back to SP
  LayerNorm...       (B, S, h)             (per TP layout)       (S/C per GPU)

Context parallelism (ring)

  Each GPU keeps ~S/C query tokens; K/V blocks move on a ring.
  Peak score tile (S/C) x (S/C) per step, not S x S on one device.
```


### Ring Attention (High Level)

With $C$ ranks in a ring, each rank starts with its local $(K_r, V_r)$. Over $C$ **rounds**, blocks rotate so every rank sees every $(K, V)$ block exactly once. At each round, the rank updates its partial attention output for its $Q_r$ using the current block.

Causal masking: blocks that correspond to **future** tokens are masked out for the current $Q$ chunk; the walkthrough in [context_parallelism_concrete_walkthrough.md](context-parallelism/context_parallelism_concrete_walkthrough.md) steps through which blocks are valid at each rotation step.

**Why not only split $Q$?** Late tokens in a causal model attend to **many** keys; if you assign **contiguous** token ranges to ranks, the **last** rank does the most work (load imbalance). Practical systems use **reordered** token-to-rank maps (for example **zig-zag** assignment) so each rank gets a mix of early and late positions and similar flop counts. See sections 8-9 of the concrete walkthrough.

**Figure (ASCII).** $C=4$ ranks on a ring: each step, every GPU passes its current $K/V$ **block** to the next rank (mod $C$). After enough steps each rank has seen every block. Planned export: `cp-kv-ring.png` ([Figures](#figures-planned-drawio-exports)).

```
  GPU0 ----KV----> GPU1 ----KV----> GPU2 ----KV----> GPU3
    ^                                                |
    +------------------- KV -------------------------+

  One rank in one step:  Q_local @ K_block^T  ->  scores (S/C)x(S/C),
                         merge into running (m, l, o) for online softmax,
                         then pass K/V block along the ring.
```


### Online Softmax Across Blocks

$\mathrm{softmax}$ normalizes over the **key** axis. For one query row, let $s \in \mathbb{R}^T$ be scores against $T$ keys and $V \in \mathbb{R}^{T \times d}$ the value vectors. The attention output is

$$
a = \sum_{j=1}^{T} \frac{e^{s_j}}{\sum_{k=1}^{T} e^{s_k}} \, V_j .
$$

**Numerical stability:** With $m = \max_j s_j$, the same weights can be written as $e^{s_j - m} / \sum_k e^{s_k - m}$, so all exponentials are $\le 1$.

**Problem:** In ring CP (or Flash-style tiling), you never store the full $s \in \mathbb{R}^T$ at once. You see **blocks** of columns of scores $s^{(1)}, s^{(2)}, \ldots$ and matching $V^{(1)}, V^{(2)}, \ldots$.

**Figure (ASCII).** Key axis split into blocks; softmax is over the **full** axis, but you only materialize one block of scores at a time. Planned export: `cp-online-softmax-blocks.png` ([Figures](#figures-planned-drawio-exports)).

```
Key positions:  [ block 0 ][ block 1 ][ block 2 ] ... [ block C-1 ]
                     |            |            |
                     v            v            v
scores seen:      QK^T_0       QK^T_1       QK^T_2   ...   (never concat to full row in memory)
                     \            |            /
                      merge (m, l, o) online softmax
                                    |
                                    v
                              final output a
```

**Idea:** Keep three running quantities for that query row (per head, batched implementations vectorize over rows):

| Symbol | Meaning after processing some key blocks |
|--------|------------------------------------------|
| $m$ | Running maximum of **all** scores seen so far |
| $\ell$ | $\sum_{j \in \text{seen}} e^{s_j - m}$ (sum of exponentials **relative to current** $m$) |
| $o$ | $\sum_{j \in \text{seen}} e^{s_j - m} \, V_j$ (unnormalized weighted sum of values at **current** $m$) |

When you finish all blocks, $a = o / \ell$. That equals the usual softmax-weighted sum of $V$ because the $e^{-m}$ factor cancels between numerator and denominator.

**Merging a new block:** Suppose the new block has scores $s'$ (vector over keys in that block) and values $V'$. Let $m' = \max(m, \max(s'))$. Everything accumulated under the old $m$ must be rescaled because the reference maximum increased:

$$
e^{s_j - m'} = e^{s_j - m} \cdot e^{m - m'} \quad \text{for } j \text{ already seen.}
$$

So

$$
\ell_{\mathrm{new}} = \ell \cdot e^{m - m'} + \sum_{i \in \mathrm{block}} e^{s'_i - m'},
$$

$$
o_{\mathrm{new}} = o \cdot e^{m - m'} + \sum_{i \in \mathrm{block}} e^{s'_i - m'} \, V'_i,
$$

then set $m \leftarrow m'$, $\ell \leftarrow \ell_{\mathrm{new}}$, $o \leftarrow o_{\mathrm{new}}$. Repeat for each block. Order matches the key order along the softmax axis (causal masks simply zero out forbidden positions inside $s'$ before this update).

**Why this is exact:** At any time, $o = \sum_{j \in \mathrm{seen}} e^{s_j - m} V_j$ and $\ell = \sum_{j \in \mathrm{seen}} e^{s_j - m}$. When $m$ is the global maximum over **seen** keys only, this is not yet the final softmax. After the **last** block, $m$ is the max over **all** keys, $\ell = \sum_k e^{s_k - m}$, and $o = \sum_k e^{s_k - m} V_k$, so $o/\ell = \sum_k \mathrm{softmax}(s)_k V_k$.

Same algebra as block-wise attention on one GPU; ring CP just delivers each $(s', V')$ block from another rank instead of from SRAM.

**Pseudocode** (one query row; real code batches $B \times H$ rows and uses masked scores):

```python
# Pseudocode: merge attention partials across K/V blocks (online softmax)
m = float("-inf")  # running max over all keys seen so far
l = 0.0            # sum_j exp(s_j - m) over keys seen so far
o = 0.0            # sum_j exp(s_j - m) * V_j  (same m as l)

for K_block, V_block in kv_blocks:  # from ring or local tiles
    scores = (Q @ K_block.transpose(-2, -1)) / (d ** 0.5)  # apply causal mask here
    m_new = max(m, scores.max())  # global max over old keys + this block
    scale = exp(m - m_new)        # rescale previous partials
    p = exp(scores - m_new)       # unnormalized weights for this block
    l = l * scale + p.sum()
    o = o * scale + p @ V_block
    m = m_new

out = o / l
```

Production kernels (FlashAttention, fused ring attention) use the same recurrence in lower precision with careful ordering to limit drift; they also **overlap** the next $K/V$ receive with the matmul that forms `scores`. The **mathematics** is the rescaling update above.


### Memory Scaling (Order of Magnitude)

Per layer, naive attention score storage is proportional to $B \cdot H \cdot S^2$ (in reduced precision). Doubling $S$ **quadruples** that term.

With context-parallel degree $C$:

- **No CP:** peak score tile **$O(S^2)$** per GPU (times $B$, $H$, and TP head sharding as in [sequence-parallelism.md](sequence-parallelism.md)).
- **Naive CP:** peak score tile **$O(S^2 / C)$** per GPU: **$(S/C) \times S$** when full $K$ is resident ([step2](context-parallelism/src/step2_cp_comparison.py)).
- **Ring attention:** peak score tile **$O(S^2 / C^2)$** per step: **$(S/C) \times (S/C)$**; the **$(S/C) \times S$** result is never stored as one tensor.

Exact constants depend on head sharding, TP, recompute, and kernel fusion. Example ballparks for large $S$ appear in [context_parallelism_concrete_walkthrough.md](context-parallelism/context_parallelism_concrete_walkthrough.md).


### Communication and the Ultra-Scale Lens

[ultra_blog.md](ultra_blog.md) stresses three pressures: **memory**, **compute efficiency**, and **communication**. Context parallelism:

- **Reduces** peak attention-related **memory** (the main win for long context).
- **Adds** communication rounds proportional to the CP degree (ring: $O(C)$ steps per layer per attention pass; can be overlapped).
- **Compute** is similar to a single-device forward in total flops (same attention), but **load balance** across ranks matters (zig-zag or similar).

Whether CP is worth it depends on whether $S$ is large enough that attention memory (or bandwidth to materialize full scores) dominates over the extra P2P / collective cost.


### Composability (Typical Training Stacks)

In large language model training you often see **data parallel** replicas of a **sharded** model: **TP** within a node or a small group, **pipeline parallel** across stages, **CP** (or **SP** under other naming schemes) for sequence, and **FSDP / ZeRO** for optimizer state. The exact **product** of parallel degrees must match hardware count and divisibility constraints ($S$ divisible by CP, heads divisible by TP, etc.). CP is the tool you reach for when **context length**, not parameter count alone, blows the memory budget.


### Code in This Repository

| Artifact | Role |
|----------|------|
| [context-parallelism/src/step1_test_nccl.py](context-parallelism/src/step1_test_nccl.py) | Sanity-check `torch.distributed` + NCCL (`LOCAL_RANK`, `device_id` init) |
| [context-parallelism/src/step2_cp_comparison.py](context-parallelism/src/step2_cp_comparison.py) | **Naive CP:** compares full $(S,S)$ scores vs **$(S/C) \times S$** logits per rank (full $K,V$); not ring $(S/C)^2$ tiles |
| [context-parallelism/src/step3_plot.py](context-parallelism/src/step3_plot.py) | Plots JSON output from step 2 |
| [context-parallelism/context_parallelism_concrete_walkthrough.md](context-parallelism/context_parallelism_concrete_walkthrough.md) | Step-by-step ring + online softmax + zig-zag load balance |
| [context-parallelism/README.md](context-parallelism/README.md) | Quick commands and expected stdout shape |

From `context-parallelism/` (after `uv`/`torch` env is set up as in [developer.md](developer.md)):

```bash
torchrun --nproc_per_node=2 src/step1_test_nccl.py
torchrun --nproc_per_node=2 src/step2_cp_comparison.py
python3 src/step3_plot.py
```

Step 2 writes `outputs/cp_results.json` in the current working directory; run step 3 from the same directory or pass the path your plot script expects.


### When to Use Context Parallelism

**Use CP** when you need **long context** and attention **$O(S^2)$** memory (or bandwidth from writing full scores) is the limiter, and you have enough GPUs / interconnect to absorb ring or block communication.

**Less critical** when $S$ is short (e.g. a few thousand tokens) and TP + SP + batch tuning already fit; CP adds complexity and synchronization surface area.

**Practical constraints:** $S$ (and microbatch shapes) must be divisible by CP degree; causal masking and load balancing require careful layout; inference and training wrappers differ (KV cache sharding for multi-turn generation is its own design space).


### Figures (planned draw.io exports)

Export from draw.io (or another tool) into `context-parallelism/images/` using the names below. Then paste the **embed line** immediately **above** the matching **Figure (ASCII)** block in this file so readers see the vector figure first; keep the ASCII as a text fallback for diffs and quick edits.

Paths are relative to the **repository root** (same style as [sequence-parallelism.md](sequence-parallelism.md) images).

| Topic | Suggested file | Embed line (after file exists) | ASCII figure location |
|-------|----------------|-------------------------------|------------------------|
| Full causal $S \times S$ score grid | `context-parallelism/images/cp-attention-score-sxs.png` | `![Attention scores S by S](context-parallelism/images/cp-attention-score-sxs.png)` | [The Bottleneck TP and SP Leave Behind](#the-bottleneck-tp-and-sp-leave-behind) |
| Naive $(S/C) \times S$ vs ring $(S/C) \times (S/C)$ tiles | `context-parallelism/images/cp-naive-vs-ring-tiles.png` | `![Naive CP vs ring score tiles](context-parallelism/images/cp-naive-vs-ring-tiles.png)` | [Naive CP vs ring attention](#naive-cp-vs-ring-attention-peak-score-tile) |
| Megatron SP+TP attention vs CP ring (flow) | `context-parallelism/images/cp-megatron-sp-vs-cp-flow.png` | `![SP plus TP vs context parallel attention](context-parallelism/images/cp-megatron-sp-vs-cp-flow.png)` | [Sequence Parallelism vs Context Parallelism](#sequence-parallelism-vs-context-parallelism) |
| $K/V$ block ring across GPUs | `context-parallelism/images/cp-kv-ring.png` | `![KV block ring](context-parallelism/images/cp-kv-ring.png)` | [Ring Attention (High Level)](#ring-attention-high-level) |
| Online softmax: key blocks into $(m,\ell,o)$ | `context-parallelism/images/cp-online-softmax-blocks.png` | `![Online softmax over blocks](context-parallelism/images/cp-online-softmax-blocks.png)` | [Online Softmax Across Blocks](#online-softmax-across-blocks) |
| (Optional) Zig-zag vs contiguous token layout | `context-parallelism/images/cp-zigzag-token-layout.png` | `![Zig-zag token assignment](context-parallelism/images/cp-zigzag-token-layout.png)` | Not in ASCII yet; pairs with [concrete walkthrough](context-parallelism/context_parallelism_concrete_walkthrough.md) sections 8-9 |

**Naming:** Prefer `cp-` prefix and kebab-case so these sit next to [cp_comparison.png](context-parallelism/images/cp_comparison.png) without clashes. Use `.svg` instead of `.png` if you prefer; update the embed path accordingly.


### Further Reading

- Ring attention and blockwise attention: see papers and implementations referenced from the Hugging Face **Ultra-Scale Playbook** and educational repos such as **picotron** / **nanotron** ([ultra_blog.md](ultra_blog.md) pointers).
- This repo's long-form narrative: [context_parallelism_concrete_walkthrough.md](context-parallelism/context_parallelism_concrete_walkthrough.md).


See [developer.md](developer.md) for full setup and CLI flags.