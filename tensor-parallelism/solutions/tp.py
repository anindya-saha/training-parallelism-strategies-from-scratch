"""Tensor Parallelism -- implemented from scratch.

This module contains the core TP building blocks:

Autograd communication primitives:
    _CopyToParallelRegion    -- identity forward, all-reduce backward
    _ReduceFromParallelRegion -- all-reduce forward, identity backward
    _AllGatherForTP          -- all-gather (last dim) forward, chunk backward

Sequence parallelism primitives (Megatron-LM v3):
    _AllGatherFromSP         -- all-gather (seq dim) forward, reduce-scatter backward
    _ReduceScatterToSP       -- reduce-scatter (seq dim) forward, all-gather backward
    _ScatterToSP             -- scatter (seq dim) forward, all-gather backward

TP linear layers:
    ColumnParallelLinear     -- splits output dim across GPUs
    RowParallelLinear        -- splits input dim across GPUs

TP model components:
    TPAttention              -- attention split by heads (supports GQA)
    TPFFN                    -- FFN split: col-parallel W1, row-parallel W2
    TPTransformerBlock       -- LN + TPAttention + TPFFN (supports SP)
    TPGPT                    -- full GPT with TP + optional SP

Loss:
    tp_cross_entropy         -- cross-entropy over TP-split logits

Communication per transformer block:
    Without SP:  2 all-reduces (1 in attn W_o, 1 in FFN W2)
    With SP:     2 all-gather + 2 reduce-scatter (same total volume)
                 But LN/residual activations are sequence-split -> less memory

All layers accept an optional process_group for 2D parallelism (TP + FSDP).
"""

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from model import (
    D_FF,
    D_HEAD,
    D_MODEL,
    MAX_SEQ_LEN,
    ModelConfig,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    SMALL_CONFIG,
    VOCAB_SIZE,
    count_parameters,
    get_gpu_memory_mb,
    get_gpu_peak_memory_mb,
    repeat_kv,
)


# ---------------------------------------------------------------------------
# Process group helper
# ---------------------------------------------------------------------------


def _get_pg(pg):
    """Return the given process group, or the default if None."""
    if pg is not None:
        return pg
    return dist.distributed_c10d._get_default_group()


# ---------------------------------------------------------------------------
# Autograd functions for TP communication
# ---------------------------------------------------------------------------


class _CopyToParallelRegion(torch.autograd.Function):
    """Identity in forward, all-reduce in backward.

    Placed BEFORE column-parallel layers so that during backprop the
    gradient (which arrives split) is summed across ranks.
    """

    @staticmethod
    def forward(ctx, x, process_group):
        ctx.process_group = process_group
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=_get_pg(ctx.process_group))
        return grad, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce in forward, identity in backward.

    Placed AFTER row-parallel layers to combine partial sums across ranks.
    """

    @staticmethod
    def forward(ctx, x, process_group):
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=_get_pg(process_group))
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _AllGatherForTP(torch.autograd.Function):
    """All-gather last dim in forward, chunk/scatter in backward.

    Used for TP cross-entropy: reconstructs full vocab logits.
    """

    @staticmethod
    def forward(ctx, x, process_group):
        pg = _get_pg(process_group)
        ws = dist.get_world_size(pg)
        ctx.ws = ws
        ctx.process_group = process_group
        gathered = [torch.zeros_like(x) for _ in range(ws)]
        dist.all_gather(gathered, x.contiguous(), group=pg)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        pg = _get_pg(ctx.process_group)
        rank = dist.get_rank(pg)
        chunks = grad.chunk(ctx.ws, dim=-1)
        return chunks[rank].contiguous(), None


# ---------------------------------------------------------------------------
# Sequence Parallelism autograd functions
# ---------------------------------------------------------------------------
# SP splits activations along the sequence dimension (dim=1) between TP
# regions.  LayerNorm and residual connections operate on (B, T/N, d)
# instead of (B, T, d), reducing activation memory by a factor of N.


class _AllGatherFromSP(torch.autograd.Function):
    """All-gather seq dim in forward, reduce-scatter in backward.

    (B, T/N, d) -> (B, T, d)  [forward]
    (B, T, d)   -> (B, T/N, d) via reduce-scatter [backward]
    """

    @staticmethod
    def forward(ctx, x, process_group):
        pg = _get_pg(process_group)
        ctx.process_group = process_group
        gathered = [torch.zeros_like(x) for _ in range(dist.get_world_size(pg))]
        dist.all_gather(gathered, x.contiguous(), group=pg)
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(ctx, grad):
        pg = _get_pg(ctx.process_group)
        ws = dist.get_world_size(pg)
        input_list = [c.contiguous() for c in grad.chunk(ws, dim=1)]
        output = torch.zeros_like(input_list[0])
        dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=pg)
        return output, None


class _ReduceScatterToSP(torch.autograd.Function):
    """Reduce-scatter seq dim in forward, all-gather in backward.

    (B, T, d) partial sums -> (B, T/N, d) complete [forward]
    (B, T/N, d) -> (B, T, d) via all-gather        [backward]
    """

    @staticmethod
    def forward(ctx, x, process_group):
        pg = _get_pg(process_group)
        ctx.process_group = process_group
        ws = dist.get_world_size(pg)
        input_list = [c.contiguous() for c in x.chunk(ws, dim=1)]
        output = torch.zeros_like(input_list[0])
        dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=pg)
        return output

    @staticmethod
    def backward(ctx, grad):
        pg = _get_pg(ctx.process_group)
        gathered = [torch.zeros_like(grad) for _ in range(dist.get_world_size(pg))]
        dist.all_gather(gathered, grad.contiguous(), group=pg)
        return torch.cat(gathered, dim=1), None


class _ScatterToSP(torch.autograd.Function):
    """Scatter seq dim in forward (take this rank's chunk), all-gather in backward.

    (B, T, d) replicated -> (B, T/N, d) local chunk [forward]
    (B, T/N, d)           -> (B, T, d) all-gather    [backward]
    """

    @staticmethod
    def forward(ctx, x, process_group):
        pg = _get_pg(process_group)
        ws = dist.get_world_size(pg)
        rank = dist.get_rank(pg)
        ctx.process_group = process_group
        T = x.shape[1]
        assert T % ws == 0, f"seq_len={T} not divisible by world_size={ws}"
        chunk_size = T // ws
        return x[:, rank * chunk_size : (rank + 1) * chunk_size, :].contiguous()

    @staticmethod
    def backward(ctx, grad):
        pg = _get_pg(ctx.process_group)
        gathered = [torch.zeros_like(grad) for _ in range(dist.get_world_size(pg))]
        dist.all_gather(gathered, grad.contiguous(), group=pg)
        return torch.cat(gathered, dim=1), None


# ---------------------------------------------------------------------------
# Column-Parallel Linear
# ---------------------------------------------------------------------------
# Full weight: (d_out, d_in) -> this GPU: (d_out // N, d_in)
# Input: same on all GPUs    -> output: each GPU has a different slice
# NO communication in forward (but _CopyToParallelRegion adds all-reduce
# in backward to accumulate gradient partials).


class ColumnParallelLinear(nn.Module):
    """Linear layer with output dimension split across GPUs.

    Args:
        input_is_parallel: if True, skip the _CopyToParallelRegion wrapper.
            Use when the caller has already gathered the input (SP attention
            does a single all-gather before Q/K/V projections).
        process_group: optional process group for 2D parallelism.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True,
                 input_is_parallel: bool = False, process_group=None):
        super().__init__()
        self.input_is_parallel = input_is_parallel
        self.process_group = process_group
        pg = _get_pg(process_group)
        ws = dist.get_world_size(pg)
        assert d_out % ws == 0, f"d_out={d_out} not divisible by world_size={ws}"
        self.d_out_local = d_out // ws
        self.weight = nn.Parameter(torch.empty(self.d_out_local, d_in))
        self.bias = (
            nn.Parameter(torch.empty(self.d_out_local)) if bias else None
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(d_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            x = _CopyToParallelRegion.apply(x, self.process_group)
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Row-Parallel Linear
# ---------------------------------------------------------------------------
# Full weight: (d_out, d_in) -> this GPU: (d_out, d_in // N)
# Input: local slice         -> output: partial sum, needs ALL-REDUCE
# With sequence_parallel=True, uses reduce-scatter instead of all-reduce.


class RowParallelLinear(nn.Module):
    """Linear layer with input dimension split across GPUs.

    Args:
        sequence_parallel: if True, output is reduce-scattered along the
            sequence dimension instead of all-reduced.
        process_group: optional process group for 2D parallelism.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True,
                 sequence_parallel: bool = False, process_group=None):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.process_group = process_group
        pg = _get_pg(process_group)
        ws = dist.get_world_size(pg)
        rank = dist.get_rank(pg)
        assert d_in % ws == 0, f"d_in={d_in} not divisible by world_size={ws}"
        self.d_in_local = d_in // ws
        self.weight = nn.Parameter(torch.empty(d_out, self.d_in_local))
        self.bias = None
        if bias and rank == 0:
            self.bias = nn.Parameter(torch.empty(d_out))
            bound = 1 / math.sqrt(d_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.sequence_parallel:
            return _ReduceScatterToSP.apply(out, self.process_group)
        return _ReduceFromParallelRegion.apply(out, self.process_group)


# ---------------------------------------------------------------------------
# TP Attention -- heads split across GPUs, with GQA support
# ---------------------------------------------------------------------------
# GQA: n_kv_heads < n_heads.  K/V projections are smaller.  Each GPU holds
# n_heads_local Q heads and n_kv_heads_local KV heads.  KV heads are repeated
# n_rep times before the attention dot-product.
#
# SP mode: a single all-gather is done at the input (before Q/K/V), and W_o
# uses reduce-scatter at the output.  This avoids redundant all-gathers for
# each of Q, K, V separately.


class TPAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 n_kv_heads: Optional[int] = None,
                 sequence_parallel: bool = False, process_group=None):
        super().__init__()
        kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        pg = _get_pg(process_group)
        ws = dist.get_world_size(pg)
        assert n_heads % ws == 0, f"n_heads={n_heads} not divisible by ws={ws}"
        assert kv_heads % ws == 0, f"n_kv_heads={kv_heads} not divisible by ws={ws}"

        self.n_heads_local = n_heads // ws
        self.n_kv_heads_local = kv_heads // ws
        self.n_rep = n_heads // kv_heads
        self.d_head = d_model // n_heads
        self.d_local = self.n_heads_local * self.d_head
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.sequence_parallel = sequence_parallel
        self.process_group = process_group

        d_q = n_heads * self.d_head
        d_kv = kv_heads * self.d_head

        # In SP mode, we all-gather once before all projections, so the
        # individual col-parallel layers skip their internal comm wrapper.
        ip = sequence_parallel
        self.W_q = ColumnParallelLinear(
            d_model, d_q, bias=False,
            input_is_parallel=ip, process_group=process_group,
        )
        self.W_k = ColumnParallelLinear(
            d_model, d_kv, bias=False,
            input_is_parallel=ip, process_group=process_group,
        )
        self.W_v = ColumnParallelLinear(
            d_model, d_kv, bias=False,
            input_is_parallel=ip, process_group=process_group,
        )
        self.W_o = RowParallelLinear(
            d_q, d_model, bias=False,
            sequence_parallel=sequence_parallel, process_group=process_group,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel:
            x = _AllGatherFromSP.apply(x, self.process_group)

        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads_local, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_kv_heads_local, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads_local, self.d_head).transpose(1, 2)

        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_local)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# TP FFN -- col-parallel W1, row-parallel W2
# ---------------------------------------------------------------------------


class TPFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 sequence_parallel: bool = False, process_group=None):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.process_group = process_group
        ip = sequence_parallel
        self.W1 = ColumnParallelLinear(
            d_model, d_ff,
            input_is_parallel=ip, process_group=process_group,
        )
        self.W2 = RowParallelLinear(
            d_ff, d_model,
            sequence_parallel=sequence_parallel, process_group=process_group,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel:
            x = _AllGatherFromSP.apply(x, self.process_group)
        return self.W2(F.gelu(self.W1(x)))


# ---------------------------------------------------------------------------
# TP Transformer Block & Full Model
# ---------------------------------------------------------------------------


class TPTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_kv_heads: Optional[int] = None,
                 sequence_parallel: bool = False,
                 process_group=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TPAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads,
            sequence_parallel=sequence_parallel, process_group=process_group,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = TPFFN(
            d_model, d_ff,
            sequence_parallel=sequence_parallel, process_group=process_group,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TPGPT(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None,
                 sequence_parallel: bool = False, process_group=None):
        super().__init__()
        if config is None:
            config = SMALL_CONFIG
        c = config
        self.config = c
        self.sequence_parallel = sequence_parallel
        self.process_group = process_group

        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_emb = nn.Embedding(c.max_seq_len, c.d_model)
        self.blocks = nn.ModuleList([
            TPTransformerBlock(
                c.d_model, c.n_heads, c.d_ff,
                n_kv_heads=c.n_kv_heads,
                sequence_parallel=sequence_parallel,
                process_group=process_group,
            )
            for _ in range(c.n_layers)
        ])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.lm_head = ColumnParallelLinear(
            c.d_model, c.vocab_size, bias=False,
            input_is_parallel=sequence_parallel,
            process_group=process_group,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        if self.sequence_parallel:
            x = _ScatterToSP.apply(x, self.process_group)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if self.sequence_parallel:
            x = _AllGatherFromSP.apply(x, self.process_group)

        return self.lm_head(x)


# ---------------------------------------------------------------------------
# TP-aware cross-entropy loss
# ---------------------------------------------------------------------------


def tp_cross_entropy(
    logits_local: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int = VOCAB_SIZE,
    process_group=None,
) -> torch.Tensor:
    """Cross-entropy over TP-split logits.

    Each GPU holds logits for a slice of the vocabulary. We all-gather the
    full logits before computing the loss so gradients flow correctly.
    """
    full_logits = _AllGatherForTP.apply(logits_local, process_group)
    return F.cross_entropy(full_logits.view(-1, vocab_size), labels.view(-1))
