# Context Parallelism: A Concrete Walkthrough with Actual Numbers

---

## 1. The Problem: Why We Need Context Parallelism

In our previous walkthroughs, we saw how Tensor Parallelism (TP) splits model weights and Sequence Parallelism (SP) reduces activation memory for LayerNorm/Dropout/Residuals. But there's still a major bottleneck:

```
The Attention Score Matrix: (B, H, S, S)

This is O(S²) — it EXPLODES with sequence length!

Sequence Length    Attention Memory (per layer, B=1, H=32)
─────────────────────────────────────────────────────────────
    1,024          1K × 1K × 32 × 2 bytes    =    64 MB
    4,096          4K × 4K × 32 × 2 bytes    =     1 GB
   16,384         16K × 16K × 32 × 2 bytes   =    16 GB
   65,536         64K × 64K × 32 × 2 bytes   =   256 GB  ← IMPOSSIBLE!
  131,072        128K × 128K × 32 × 2 bytes  = 1,024 GB  ← WAY IMPOSSIBLE!
```

**Neither TP nor SP help here:**
- TP splits heads: `(B, H/N, S, S)` — still O(S²) per GPU
- SP doesn't touch attention (it's in the TP region)

**Context Parallelism (CP)** solves this by splitting the sequence dimension **inside the attention computation itself**.

---

## 2. Our Setup for This Walkthrough

```
Sequence length (S)         = 16 tokens (easily divisible)
Number of attention heads   = 4
Head dimension (d_k)        = 4
Number of GPUs (CP degree)  = 4

Each GPU will handle S/CP = 16/4 = 4 tokens
```

We'll trace through how 4 GPUs collaboratively compute attention over 16 tokens.

---

## 3. The Core Idea: Split Sequence Across the ENTIRE Model

### Sequence Parallelism (SP) vs Context Parallelism (CP):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SEQUENCE PARALLELISM (SP):                                                 │
│    • Splits sequence only in "SP regions" (LayerNorm, Dropout, Residuals)  │
│    • Inside TP region (attention, FFN): sequence is FULL                    │
│    • Attention computes full (S × S) matrix                                 │
│                                                                             │
│  CONTEXT PARALLELISM (CP):                                                  │
│    • Splits sequence across the ENTIRE model, including attention          │
│    • Each GPU only has S/CP tokens                                          │
│    • Attention requires communication to get K/V from other GPUs           │
│    • Uses Ring Attention for efficient K/V exchange                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Each GPU Holds (CP=4, S=16):

```
GPU 0: tokens  0- 3  →  Q₀, K₀, V₀
GPU 1: tokens  4- 7  →  Q₁, K₁, V₁
GPU 2: tokens  8-11  →  Q₂, K₂, V₂
GPU 3: tokens 12-15  →  Q₃, K₃, V₃
```

---

## 4. The Attention Problem: Each Query Needs ALL Keys

For causal attention, query at position `i` attends to keys at positions `0..i`:

```
Query Position    Needs Keys From
──────────────────────────────────
     0            {0}
     1            {0, 1}
     2            {0, 1, 2}
     ...
     7            {0, 1, 2, 3, 4, 5, 6, 7}
     ...
    15            {0, 1, 2, ..., 15}  ← needs ALL keys!
```

**The problem:** GPU 3 has queries for tokens 12-15, but needs keys from tokens 0-11 which are on GPUs 0, 1, 2!

```
GPU 3's queries (tokens 12-15) need:
  • K₀, V₀ from GPU 0  (tokens 0-3)
  • K₁, V₁ from GPU 1  (tokens 4-7)
  • K₂, V₂ from GPU 2  (tokens 8-11)
  • K₃, V₃ from itself (tokens 12-15)
```

---

## 5. Ring Attention: The Efficient Solution

Instead of all-gathering all K/V to every GPU (expensive!), we pass K/V around in a **ring**:

### The Ring Communication Pattern:

```
        ┌──────────────────────────────────────────┐
        │                                          │
        ▼                                          │
    ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐
    │ GPU 0 │ ───► │ GPU 1 │ ───► │ GPU 2 │ ───► │ GPU 3 │
    │ K₀,V₀ │      │ K₁,V₁ │      │ K₂,V₂ │      │ K₃,V₃ │
    └───────┘      └───────┘      └───────┘      └───────┘
        │                                          ▲
        │                                          │
        └──────────────────────────────────────────┘
        
Each GPU sends its K/V to the next GPU in the ring.
After CP steps, every GPU has seen all K/V pairs.
```

### Step-by-Step Ring Attention (4 GPUs, 4 Steps):

```
═══════════════════════════════════════════════════════════════════════════════
STEP 0: Each GPU computes attention with its LOCAL K/V
═══════════════════════════════════════════════════════════════════════════════
  GPU 0: Attn(Q₀, K₀, V₀)  →  partial output for tokens 0-3
  GPU 1: Attn(Q₁, K₁, V₁)  →  partial output for tokens 4-7
  GPU 2: Attn(Q₂, K₂, V₂)  →  partial output for tokens 8-11
  GPU 3: Attn(Q₃, K₃, V₃)  →  partial output for tokens 12-15
  
  SEND: GPU 0 → GPU 1 (K₀,V₀), GPU 1 → GPU 2 (K₁,V₁), etc.

═══════════════════════════════════════════════════════════════════════════════
STEP 1: Each GPU receives K/V from previous GPU, computes more attention
═══════════════════════════════════════════════════════════════════════════════
  GPU 0: receives K₃,V₃ from GPU 3  →  Attn(Q₀, K₃, V₃)  [masked out for causal!]
  GPU 1: receives K₀,V₀ from GPU 0  →  Attn(Q₁, K₀, V₀)  →  update partial
  GPU 2: receives K₁,V₁ from GPU 1  →  Attn(Q₂, K₁, V₁)  →  update partial
  GPU 3: receives K₂,V₂ from GPU 2  →  Attn(Q₃, K₂, V₂)  →  update partial
  
  SEND: rotate K/V again

═══════════════════════════════════════════════════════════════════════════════
STEP 2: Continue rotating
═══════════════════════════════════════════════════════════════════════════════
  GPU 0: receives K₂,V₂  →  Attn(Q₀, K₂, V₂)  [masked out for causal!]
  GPU 1: receives K₃,V₃  →  Attn(Q₁, K₃, V₃)  [masked out for causal!]
  GPU 2: receives K₀,V₀  →  Attn(Q₂, K₀, V₀)  →  update partial
  GPU 3: receives K₁,V₁  →  Attn(Q₃, K₁, V₁)  →  update partial
  
  SEND: rotate K/V again

═══════════════════════════════════════════════════════════════════════════════
STEP 3: Final rotation
═══════════════════════════════════════════════════════════════════════════════
  GPU 0: receives K₁,V₁  →  Attn(Q₀, K₁, V₁)  [masked out for causal!]
  GPU 1: receives K₂,V₂  →  Attn(Q₁, K₂, V₂)  [masked out for causal!]
  GPU 2: receives K₃,V₃  →  Attn(Q₂, K₃, V₃)  [masked out for causal!]
  GPU 3: receives K₀,V₀  →  Attn(Q₃, K₀, V₀)  →  update partial
  
  DONE! Each GPU now has complete attention output for its tokens.
```

---

## 6. The Online Softmax Trick: How to Combine Partial Attentions

Here's the tricky part: attention uses softmax, which requires knowing ALL values to compute the normalization. How do we combine partial attentions computed at different steps?

### Standard Softmax (requires all values):

```
Attention(Q, K, V) = softmax(QK^T / √d) · V

softmax(x)_i = exp(x_i) / Σⱼ exp(x_j)  ← needs the FULL sum!
```

### Online Softmax (incremental computation):

We can compute softmax incrementally by tracking:
1. `m` = running maximum (for numerical stability)
2. `l` = running sum of exp(scores - m)
3. `o` = running weighted output

```python
# Initialize
m = -inf      # running max
l = 0         # running sum of exp(scores)
o = 0         # running output

# For each block of K/V received:
for K_block, V_block in ring_attention_stream:
    scores = Q @ K_block.T / sqrt(d)  # (local_seq, block_seq)
    
    # Update running max
    m_new = max(m, max(scores))
    
    # Rescale previous sums
    l = l * exp(m - m_new)
    o = o * exp(m - m_new)
    
    # Add new block contribution
    p = exp(scores - m_new)      # attention weights for this block
    l = l + sum(p)               # update running sum
    o = o + p @ V_block          # update running output
    
    m = m_new

# Final output
output = o / l
```

This is the same trick used in **Flash Attention** — but here we apply it across GPUs in a ring!

---

## 7. Concrete Numbers: Walking Through Ring Attention

Let's trace actual values through one attention head with CP=4.

### Setup:

```
Sequence: 16 tokens, split into 4 chunks of 4 tokens each
Head dimension: d = 4

GPU 0: Q₀ (4×4), K₀ (4×4), V₀ (4×4)  for tokens 0-3
GPU 1: Q₁ (4×4), K₁ (4×4), V₁ (4×4)  for tokens 4-7
GPU 2: Q₂ (4×4), K₂ (4×4), V₂ (4×4)  for tokens 8-11
GPU 3: Q₃ (4×4), K₃ (4×4), V₃ (4×4)  for tokens 12-15
```

### Attention Score Matrix (Full, for Reference):

```
The full 16×16 causal attention matrix:

        K₀(0-3)   K₁(4-7)   K₂(8-11)  K₃(12-15)
       ┌─────────┬─────────┬─────────┬─────────┐
Q₀(0-3)│  ████   │         │         │         │  GPU 0 computes this
       │  ████   │         │         │         │
       │  ████   │         │         │         │
       │  ████   │         │         │         │
       ├─────────┼─────────┼─────────┼─────────┤
Q₁(4-7)│  ████   │  ████   │         │         │  GPU 1 computes this
       │  ████   │  ████   │         │         │
       │  ████   │  ████   │         │         │
       │  ████   │  ████   │         │         │
       ├─────────┼─────────┼─────────┼─────────┤
Q₂(8-11)│ ████   │  ████   │  ████   │         │  GPU 2 computes this
       │  ████   │  ████   │  ████   │         │
       │  ████   │  ████   │  ████   │         │
       │  ████   │  ████   │  ████   │         │
       ├─────────┼─────────┼─────────┼─────────┤
Q₃(12-15)│████   │  ████   │  ████   │  ████   │  GPU 3 computes this
       │  ████   │  ████   │  ████   │  ████   │
       │  ████   │  ████   │  ████   │  ████   │
       │  ████   │  ████   │  ████   │  ████   │
       └─────────┴─────────┴─────────┴─────────┘
       
████ = valid attention (not masked)
     = masked (causal: can't attend to future)
```

### What Each GPU Computes:

```
GPU 0 (queries 0-3):
  Only needs K₀ (tokens 0-3) — all local!
  Computes: Attn(Q₀, K₀, V₀)
  No ring communication needed for valid attention.

GPU 1 (queries 4-7):
  Needs K₀ (tokens 0-3) from GPU 0
  Needs K₁ (tokens 4-7) — local
  Step 0: Attn(Q₁, K₁, V₁)  →  partial
  Step 1: Receives K₀,V₀, computes Attn(Q₁, K₀, V₀)  →  combine with online softmax

GPU 2 (queries 8-11):
  Needs K₀, K₁, K₂
  Step 0: Attn(Q₂, K₂, V₂)  →  partial
  Step 1: Receives K₁, computes Attn(Q₂, K₁, V₁)  →  update
  Step 2: Receives K₀, computes Attn(Q₂, K₀, V₀)  →  update

GPU 3 (queries 12-15):
  Needs K₀, K₁, K₂, K₃ — needs ALL K/V!
  Step 0: Attn(Q₃, K₃, V₃)  →  partial
  Step 1: Receives K₂, computes Attn(Q₃, K₂, V₂)  →  update
  Step 2: Receives K₁, computes Attn(Q₃, K₁, V₁)  →  update
  Step 3: Receives K₀, computes Attn(Q₃, K₀, V₀)  →  final
```

---

## 8. The Load Imbalance Problem

Notice something problematic in the causal attention matrix:

```
                K₀        K₁        K₂        K₃
            ┌─────────┬─────────┬─────────┬─────────┐
      Q₀    │ 4×4=16  │    0    │    0    │    0    │  Total: 16
            ├─────────┼─────────┼─────────┼─────────┤
      Q₁    │   16    │   16    │    0    │    0    │  Total: 32
            ├─────────┼─────────┼─────────┼─────────┤
      Q₂    │   16    │   16    │   16    │    0    │  Total: 48
            ├─────────┼─────────┼─────────┼─────────┤
      Q₃    │   16    │   16    │   16    │   16    │  Total: 64
            └─────────┴─────────┴─────────┴─────────┘

GPU 0: 16 attention computations
GPU 1: 32 attention computations  (2× GPU 0)
GPU 2: 48 attention computations  (3× GPU 0)
GPU 3: 64 attention computations  (4× GPU 0)  ← does 4× more work!
```

This is terrible load balancing! GPU 0 finishes quickly and sits idle while GPU 3 is still computing.

---

## 9. Zig-Zag Ring Attention: Balanced Load Distribution

### The Zig-Zag Token Assignment:

Instead of assigning tokens sequentially:
```
Sequential (BAD):
  GPU 0: tokens 0,  1,  2,  3
  GPU 1: tokens 4,  5,  6,  7
  GPU 2: tokens 8,  9, 10, 11
  GPU 3: tokens 12, 13, 14, 15
```

We interleave them in a zig-zag pattern:
```
Zig-Zag (GOOD):
  GPU 0: tokens 0,  7,  8, 15   (mix of early and late)
  GPU 1: tokens 1,  6,  9, 14
  GPU 2: tokens 2,  5, 10, 13
  GPU 3: tokens 3,  4, 11, 12
```

### Why This Balances Load:

```
Zig-Zag Pattern for 16 tokens across 4 GPUs:

Position:    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
GPU:         0  1  2  3  3  2  1  0  0  1  2  3  3  2  1  0

The pattern: 0,1,2,3,3,2,1,0,0,1,2,3,3,2,1,0 (zig-zag!)
```

Now each GPU has a mix of early tokens (which attend to few keys) and late tokens (which attend to many keys):

```
GPU 0 has tokens: 0, 7, 8, 15
  Token  0: attends to 1 position   (itself)
  Token  7: attends to 8 positions  (0-7)
  Token  8: attends to 9 positions  (0-8)
  Token 15: attends to 16 positions (0-15)
  Total: 1 + 8 + 9 + 16 = 34

GPU 1 has tokens: 1, 6, 9, 14
  Token  1: attends to 2 positions
  Token  6: attends to 7 positions
  Token  9: attends to 10 positions
  Token 14: attends to 15 positions
  Total: 2 + 7 + 10 + 15 = 34

GPU 2 has tokens: 2, 5, 10, 13
  Total: 3 + 6 + 11 + 14 = 34

GPU 3 has tokens: 3, 4, 11, 12
  Total: 4 + 5 + 12 + 13 = 34

ALL GPUs do exactly 34 attention computations! Perfect balance!
```

### Zig-Zag Attention Mask Visualization:

```
With zig-zag assignment, the attention mask looks like:

         Keys (by GPU assignment)
         GPU0  GPU1  GPU2  GPU3  GPU3  GPU2  GPU1  GPU0  ...
Queries  0     1     2     3     4     5     6     7     ...
─────────────────────────────────────────────────────────────
GPU0 (0) ██
GPU1 (1) ██    ██
GPU2 (2) ██    ██    ██
GPU3 (3) ██    ██    ██    ██
GPU3 (4) ██    ██    ██    ██    ██
GPU2 (5) ██    ██    ██    ██    ██    ██
GPU1 (6) ██    ██    ██    ██    ██    ██    ██
GPU0 (7) ██    ██    ██    ██    ██    ██    ██    ██
GPU0 (8) ██    ██    ██    ██    ██    ██    ██    ██    ██
...

Each GPU has roughly the same number of colored squares!
```

---

## 10. Memory Savings with Context Parallelism

### Without CP (TP=2 only):

```
Attention score matrix per GPU: (B, H/2, S, S)

For S=65536 (64K context), H=32, B=1:
  Memory = 1 × 16 × 65536 × 65536 × 2 bytes
         = 128 GB per layer per GPU  ← IMPOSSIBLE!
```

### With CP=4 (plus TP=2):

```
Each GPU only computes attention for S/CP = 65536/4 = 16384 queries

Attention score matrix per GPU: (B, H/2, S/CP, S/CP)
  = (1, 16, 16384, 16384) per ring step

But we only store one block at a time (online softmax)!
  Memory = 1 × 16 × 16384 × 16384 × 2 bytes / CP
         = 8 GB per layer per GPU  ← FITS!
```

### Memory Usage Summary (8B Model):

```
                    No Parallelism    TP=2, CP=1    TP=2, CP=4
                    ──────────────    ──────────    ──────────
Seq Length 1024         ~90 GB          ~50 GB        ~50 GB
Seq Length 4096        ~105 GB          ~60 GB        ~50 GB
Seq Length 16384       ~140 GB          ~90 GB        ~55 GB
Seq Length 65536         OOM             OOM          ~85 GB
Seq Length 131072        OOM             OOM         ~130 GB

(Values approximate, based on activation memory scaling)

The key insight: With CP, activation memory scales as O(S/CP) not O(S)!
```

---

## 11. Communication Patterns: AllGather vs Ring (All-to-All)

### Option 1: AllGather Implementation

```
All GPUs gather complete K/V before computing attention:

Step 1: AllGather K → each GPU has full K (all 16 tokens)
Step 2: AllGather V → each GPU has full V (all 16 tokens)
Step 3: Each GPU computes attention for its queries

Communication: 2 × AllGather of size (S, d_head) per head
Memory: Must store full K, V temporarily → O(S) extra memory
```

```
Timeline:
GPU 0: [──AllGather K──][──AllGather V──][─────Compute Attention─────]
GPU 1: [──AllGather K──][──AllGather V──][─────Compute Attention─────]
GPU 2: [──AllGather K──][──AllGather V──][─────Compute Attention─────]
GPU 3: [──AllGather K──][──AllGather V──][─────Compute Attention─────]
                                          
Pros: Simple implementation
Cons: High memory (full K,V), communication not overlapped with compute
```

### Option 2: Ring (All-to-All) Implementation

```
K/V passed around ring, computation overlapped:

Each step:
  1. Async send current K/V to next GPU
  2. Compute attention with current K/V
  3. Receive K/V from previous GPU

Communication: CP rounds of P2P sends, each of size (S/CP, d_head)
Memory: Only store 2 blocks of K/V at a time → O(S/CP) extra memory
```

```
Timeline (overlapped):
GPU 0: [Compute₀][──Compute₁───][──Compute₂───][──Compute₃───]
       [Send K₀V₀→][Recv K₃V₃ ][ Send    →   ][   Recv      ]
                   [Send K₃V₃→][Recv K₂V₂]    [Send→][Recv  ]

Communication overlaps with computation!

Pros: Memory efficient, overlapped communication
Cons: More complex implementation, CP rounds of latency
```

---

## 12. Putting It All Together: CP + TP + SP

### The Full Picture for Long-Context Training:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CONTEXT PARALLELISM (CP=4)                                                 │
│    • Splits sequence across GPUs for attention                              │
│    • Ring Attention exchanges K/V efficiently                               │
│    • Reduces attention memory from O(S²) to O(S²/CP)                        │
│                                                                             │
│  Combined with TENSOR PARALLELISM (TP=8) + SEQUENCE PARALLELISM (SP=8):    │
│    • TP: splits attention heads and FFN across 8 GPUs                       │
│    • SP: splits LayerNorm/Dropout/Residuals along sequence                  │
│    • CP: splits attention computation along sequence                        │
│                                                                             │
│  Together they enable training with 128K+ token sequences!                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Through One Transformer Block:

```
Input: (B, S, h) distributed as:
  • Sequence split by CP: each GPU has S/CP tokens
  • Hidden dim split by TP: each GPU has h/TP dimensions
  • Result: each GPU has (B, S/CP, h/TP) locally

┌─────────────────────────────────────────────────────────────────────────────┐
│  1. LayerNorm (SP region)                                                   │
│     Each GPU: (B, S/CP, h) — sequence sharded, full hidden                 │
│     AllGather hidden dim for TP region                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. Q, K, V Projection (TP region)                                          │
│     Each GPU computes Q, K, V for its S/CP tokens                          │
│     Q: (B, S/CP, h/TP)                                                      │
│     K: (B, S/CP, h/TP)                                                      │
│     V: (B, S/CP, h/TP)                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. Ring Attention (CP region)                                              │
│     Ring communication exchanges K, V across CP group                       │
│     Each GPU computes attention for its S/CP queries                        │
│     Online softmax combines partial results                                 │
│     Output: (B, S/CP, h/TP)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. Output Projection (TP region)                                           │
│     Row-parallel W_o with reduce-scatter                                    │
│     Output: (B, S/CP, h) — back to SP region                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. Residual + LayerNorm + FFN + Residual (SP + TP regions)                │
│     Same as before, but sequence stays sharded by CP throughout            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Communication Costs

### Per Transformer Block:

```
Operation                      Communication Volume       When
─────────────────────────────────────────────────────────────────────────────
TP (all-gather/reduce-scatter) 4 × (B × S/CP × h)        Every layer
SP (implicit in TP)            (same as TP)              Every layer
CP (ring attention)            2 × CP × (B × S/CP × d)   Every attention
                               (K and V, CP rounds)
─────────────────────────────────────────────────────────────────────────────
```

### Example: 70B Model, S=128K, CP=8, TP=8:

```
Per attention layer:
  Ring attention comm = 2 × 8 × (B × 16K × 128) × 2 bytes
                      = 64 MB × B per layer

With 80 layers: 5.12 GB × B per forward pass for CP communication

This is significant but:
  1. Overlapped with compute (ring attention)
  2. Better than OOM without CP!
```

---

## 14. When to Use Context Parallelism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  USE CONTEXT PARALLELISM WHEN:                                              │
│                                                                             │
│  • Sequence length > 8K tokens                                              │
│  • Attention memory exceeds GPU capacity even with TP+SP                    │
│  • Training long-context models (32K, 64K, 128K+ tokens)                    │
│                                                                             │
│  TYPICAL CONFIGURATIONS:                                                    │
│                                                                             │
│  Sequence Length     Recommended CP                                         │
│  ────────────────────────────────────                                       │
│       ≤ 4K           CP=1 (not needed)                                      │
│     4K - 16K         CP=2                                                   │
│    16K - 64K         CP=4                                                   │
│    64K - 256K        CP=8                                                   │
│      > 256K          CP=16+                                                 │
│                                                                             │
│  HIERARCHY:                                                                 │
│    TP + SP (within node) → PP (across nodes) → DP (replicas) → CP (long seq)│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 15. Summary: The Complete Parallelism Toolkit

```
┌──────────────────┬────────────────────┬────────────────────┬─────────────────┐
│  TECHNIQUE       │  WHAT IT SPLITS    │  MEMORY SAVINGS    │  WHEN TO USE    │
├──────────────────┼────────────────────┼────────────────────┼─────────────────┤
│  Tensor          │  Weight matrices   │  Parameters: 1/TP  │  Model too big  │
│  Parallelism     │  (columns/rows)    │  Activations: some │  for 1 GPU      │
├──────────────────┼────────────────────┼────────────────────┼─────────────────┤
│  Sequence        │  LayerNorm/Dropout │  Activations: 1/SP │  Always use     │
│  Parallelism     │  along sequence    │  (in SP regions)   │  with TP        │
├──────────────────┼────────────────────┼────────────────────┼─────────────────┤
│  Context         │  Attention along   │  Attn memory: 1/CP │  Long sequences │
│  Parallelism     │  sequence (Q,K,V)  │  (Ring Attention)  │  (>8K tokens)   │
├──────────────────┼────────────────────┼────────────────────┼─────────────────┤
│  Pipeline        │  Layers across     │  Parameters: 1/PP  │  Model too big  │
│  Parallelism     │  nodes             │  Activations: 1/PP │  for 1 node     │
├──────────────────┼────────────────────┼────────────────────┼─────────────────┤
│  Data            │  Batch across      │  None (replicated) │  Scale          │
│  Parallelism     │  replicas          │  + ZeRO for optim  │  throughput     │
└──────────────────┴────────────────────┴────────────────────┴─────────────────┘

The complete stack for training LLaMA-70B with 128K context:
  • TP=8 + SP=8 (within node)
  • CP=4 (for 128K sequence)
  • PP=4 (across nodes)
  • DP=8 (across replicas)
  • Total: 8 × 4 × 4 × 8 = 1024 GPUs
```
