"""Tensor Parallelism -- implemented from scratch.

This module contains the core TP building blocks:

Autograd communication primitives:
    _CopyToParallelRegion    -- identity forward, all-reduce backward
    _ReduceFromParallelRegion -- all-reduce forward, identity backward
    _AllGatherForTP          -- all-gather (last dim) forward, chunk backward

TP linear layers:
    ColumnParallelLinear     -- splits output dim across GPUs
    RowParallelLinear        -- splits input dim across GPUs

TP model components:
    TPAttention              -- attention with heads split across GPUs
    TPFFN                    -- FFN split: col-parallel W1, row-parallel W2
    TPTransformerBlock       -- LN + TPAttention + TPFFN
    TPGPT                    -- full GPT with tensor parallelism

Loss:
    tp_cross_entropy         -- cross-entropy over TP-split logits

Communication per transformer block:
    2 all-reduces (1 in attn W_o, 1 in FFN W2)
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from model import (
    D_FF,
    D_HEAD,
    D_MODEL,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    VOCAB_SIZE,
    count_parameters,
    get_gpu_memory_mb,
    get_gpu_peak_memory_mb,
)


# ---------------------------------------------------------------------------
# Autograd functions for TP communication
# ---------------------------------------------------------------------------


class _CopyToParallelRegion(torch.autograd.Function):
    """Identity in forward, all-reduce in backward.

    Placed BEFORE column-parallel layers so that during backprop the
    gradient (which arrives split) is summed across ranks.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce in forward, identity in backward.

    Placed AFTER row-parallel layers to combine partial sums across ranks.
    """

    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


class _AllGatherForTP(torch.autograd.Function):
    """All-gather last dim in forward, chunk/scatter in backward.

    Used for TP cross-entropy: reconstructs full vocab logits.
    """

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


# ---------------------------------------------------------------------------
# Column-Parallel Linear
# ---------------------------------------------------------------------------
# Full weight: (d_out, d_in) -> this GPU: (d_out // N, d_in)
# Input: same on all GPUs    -> output: each GPU has a different slice
# NO communication in forward (but _CopyToParallelRegion adds all-reduce
# in backward to accumulate gradient partials).


class ColumnParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        ws = dist.get_world_size()
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
        x = _CopyToParallelRegion.apply(x)
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Row-Parallel Linear
# ---------------------------------------------------------------------------
# Full weight: (d_out, d_in) -> this GPU: (d_out, d_in // N)
# Input: local slice         -> output: partial sum, needs ALL-REDUCE


class RowParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        ws = dist.get_world_size()
        rank = dist.get_rank()
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
        return _ReduceFromParallelRegion.apply(out)


# ---------------------------------------------------------------------------
# TP Attention -- heads split across GPUs
# ---------------------------------------------------------------------------


class TPAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        ws = dist.get_world_size()
        assert n_heads % ws == 0, f"n_heads={n_heads} not divisible by ws={ws}"
        self.n_heads_local = n_heads // ws
        self.d_head = d_model // n_heads
        self.d_local = self.n_heads_local * self.d_head
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.W_q = ColumnParallelLinear(d_model, d_model, bias=False)
        self.W_k = ColumnParallelLinear(d_model, d_model, bias=False)
        self.W_v = ColumnParallelLinear(d_model, d_model, bias=False)
        self.W_o = RowParallelLinear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads_local, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads_local, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads_local, self.d_head).transpose(1, 2)

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
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = ColumnParallelLinear(d_model, d_ff)
        self.W2 = RowParallelLinear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.gelu(self.W1(x)))


# ---------------------------------------------------------------------------
# TP Transformer Block & Full Model
# ---------------------------------------------------------------------------


class TPTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TPAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = TPFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TPGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([
            TPTransformerBlock(D_MODEL, N_HEADS, D_FF)
            for _ in range(N_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = ColumnParallelLinear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# TP-aware cross-entropy loss
# ---------------------------------------------------------------------------


def tp_cross_entropy(
    logits_local: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int = VOCAB_SIZE,
) -> torch.Tensor:
    """Cross-entropy over TP-split logits.

    Each GPU holds logits for a slice of the vocabulary. We all-gather the
    full logits before computing the loss so gradients flow correctly.
    """
    full_logits = _AllGatherForTP.apply(logits_local)
    return F.cross_entropy(full_logits.view(-1, vocab_size), labels.view(-1))
