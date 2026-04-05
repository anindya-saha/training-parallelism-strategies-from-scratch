"""TP + SP GPT-style transformer model.

A minimal GPT decoder with Tensor Parallelism + Sequence Parallelism.
Weight matrices are sharded across GPUs using Column-Parallel and
Row-Parallel linear layers (same as vanilla TP).

Key differences from vanilla TP (model_gpt_tp.py):
- Row-parallel layers use REDUCE-SCATTER instead of ALL-REDUCE
  to transition from TP region (B, S, h) to SP region (B, S/N, h)
- An ALL-GATHER before column-parallel layers transitions back
  from SP region (B, S/N, h) to TP region (B, S, h)
- LayerNorm, Dropout, and Residual connections now operate on
  (B, S/N, h) instead of (B, S, h) -- saving ~1/N activation memory

Same communication volume as vanilla TP, but ~30-50% less activation memory.

Uses the default process group (all GPUs) for all communication.
"""

import argparse
import json
import logging
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from config import *

logger = logging.getLogger(__name__)


# ================================================================
#  SP communication helpers
# ================================================================


def _all_gather_along_seq(x: torch.Tensor) -> torch.Tensor:
    """Gather along sequence dim: (B, S/N, h) -> (B, S, h)."""
    ws = dist.get_world_size()
    gathered = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=1)


def _reduce_scatter_along_seq(x: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter along sequence dim: (B, S, h) -> (B, S/N, h)."""
    ws = dist.get_world_size()
    B, S, h = x.shape
    S_local = S // ws
    chunks = list(x.split(S_local, dim=1))
    output = torch.empty(B, S_local, h, device=x.device, dtype=x.dtype)
    dist.reduce_scatter(output, chunks, op=dist.ReduceOp.SUM)
    return output


# ================================================================
#  Autograd functions for TP+SP communication
# ================================================================
# _CopyToTPRegion:           identity forward,        all-reduce backward
#   -> Used before column-parallel layers (same as vanilla TP)
# _AllGatherFromSPRegion:    all-gather forward,      reduce-scatter backward   (g in Megatron diagram)
#   -> SP -> TP transition (replaces the implicit identity in vanilla TP)
# _ReduceScatterToSPRegion:  reduce-scatter forward,  all-gather backward       (g* in Megatron diagram)
#   -> TP -> SP transition (replaces ALL-REDUCE in vanilla TP)


class _CopyToTPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


class _AllGatherFromSPRegion(torch.autograd.Function):
    """Forward: all-gather (SP -> TP). Backward: reduce-scatter (TP -> SP)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return _all_gather_along_seq(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _reduce_scatter_along_seq(grad)


class _ReduceScatterToSPRegion(torch.autograd.Function):
    """Forward: reduce-scatter (TP -> SP). Backward: all-gather (SP -> TP)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return _reduce_scatter_along_seq(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _all_gather_along_seq(grad)


# ================================================================
#  Column-parallel linear
# ================================================================
# Same weight sharding as vanilla TP.
# Full: (d_out, d_in) -> this GPU: (d_out // N, d_in)
# Input: (B, S, h) after all-gather -> output: (B, S, h/N)
# No communication in forward!


class ColumnParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        ws = dist.get_world_size()
        assert d_out % ws == 0, "d_out must be divisible by world size"
        # F.linear computes x @ W.T, so W is stored as (d_out, d_in) to match PyTorch convention.
        # We shard the output dim: each GPU holds (d_out // ws, d_in).
        self.weight = nn.Parameter(torch.empty(d_out // ws, d_in))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(d_out // ws))
            nn.init.uniform_(self.bias, a=-1 / math.sqrt(d_in), b=1 / math.sqrt(d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _CopyToTPRegion.apply(x)
        return F.linear(x, self.weight, self.bias)


# ================================================================
#  Row-parallel linear (with REDUCE-SCATTER for SP)
# ================================================================
# Same weight sharding as vanilla TP.
# Full: (d_out, d_in) -> this GPU: (d_out, d_in // N)
# Input: local slice (B, S, h/N) -> output: (B, S/N, h) via reduce-scatter
#
# KEY DIFFERENCE from vanilla TP:
# - Vanilla TP uses ALL-REDUCE   -> output stays (B, S, h)
# - TP+SP uses REDUCE-SCATTER    -> output becomes (B, S/N, h)


class RowParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        ws = dist.get_world_size()
        assert d_in % ws == 0, "d_in must be divisible by world size"
        # F.linear computes x @ W.T, so W is stored as (d_out, d_in) to match PyTorch convention.
        # We shard the input dim: each GPU holds (d_out, d_in // ws).
        self.weight = nn.Parameter(torch.empty(d_out, d_in // ws))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Bias only on rank 0: reduce-scatter sums the partial results,
        # so if every rank added bias, it would be summed ws times.
        self.bias = None
        if bias and dist.get_rank() == 0:
            self.bias = nn.Parameter(torch.empty(d_out))
            nn.init.uniform_(self.bias, a=-1 / math.sqrt(d_in), b=1 / math.sqrt(d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        # REDUCE-SCATTER instead of ALL-REDUCE!
        # Transitions from TP region (B, S, h) to SP region (B, S/N, h)
        return _ReduceScatterToSPRegion.apply(out)


# ================================================================
# TP+SP attention (split by heads)
# ================================================================
# Input:  (B, S, h) after all-gather from SP region
# Output: (B, S/N, h) via reduce-scatter in W_o back to SP region


class TPAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        ws = dist.get_world_size()
        assert n_heads % ws == 0, "n_heads must be divisible by world size"

        self.d_head = d_model // n_heads
        self.n_heads_local = n_heads // ws
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.W_q = ColumnParallelLinear(d_model, d_model, bias=bias)
        self.W_k = ColumnParallelLinear(d_model, d_model, bias=bias)
        self.W_v = ColumnParallelLinear(d_model, d_model, bias=bias)
        self.W_o = RowParallelLinear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads_local, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads_local, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads_local, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = self.attn_dropout(F.softmax(attn, dim=-1))

        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)  # REDUCE-SCATTER: returns (B, S/N, h)


# ================================================================
# TP+SP FFN (feed-forward network)
# ================================================================
# Input:  (B, S, h) after all-gather from SP region
# Output: (B, S/N, h) via reduce-scatter in W2 back to SP region


class TPFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.W1 = ColumnParallelLinear(d_model, d_ff, bias=bias)
        self.W2 = RowParallelLinear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.gelu(self.W1(x)))


# ================================================================
# TP+SP GPT transformer block
# ================================================================
# LayerNorm, Dropout, and Residual connections now operate on
# (B, S/N, h) instead of (B, S, h) -- this is where memory is SAVED.
#
# Flow per sub-block:
#   SP region (B, S/N, h) -> LayerNorm -> ALL-GATHER -> TP region (B, S, h)
#   -> Attention/FFN -> REDUCE-SCATTER -> SP region (B, S/N, h) -> Dropout + Residual


class TPSPTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_bias: bool = False,
        ffn_bias: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)  # (B, S/N, h) -- memory saved
        self.attn = TPAttention(d_model, n_heads, bias=attn_bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)  # (B, S/N, h) -- memory saved
        self.ffn = TPFFN(d_model, d_ff, bias=ffn_bias)
        self.resid_dropout = nn.Dropout(dropout)  # (B, S/N, h) -- memory saved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S/N, h) -- SP region, sequence is sharded

        # --- Attention sub-block ---
        residual = x  # (B, S/N, h)
        x = _AllGatherFromSPRegion.apply(self.norm1(x))  # (B, S/N, h) -> (B, S, h)
        x = self.attn(x)  # (B, S, h) -> (B, S/N, h) via reduce-scatter
        x = residual + self.resid_dropout(x)  # (B, S/N, h)

        # --- FFN sub-block ---
        residual = x  # (B, S/N, h)
        x = _AllGatherFromSPRegion.apply(self.norm2(x))  # (B, S/N, h) -> (B, S, h)
        x = self.ffn(x)  # (B, S, h) -> (B, S/N, h) via reduce-scatter
        x = residual + self.resid_dropout(x)  # (B, S/N, h)

        return x  # (B, S/N, h) -- stays in SP region


# ================================================================
# Full TP+SP GPT model
# ================================================================


class TPSPGPT(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
        n_layers: int = N_LAYERS,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = MAX_SEQ_LEN,
        attn_bias: bool = False,
        ffn_bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ws = dist.get_world_size()
        self.rank = dist.get_rank()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)  # (B, S/N, h) -- in SP region
        self.blocks = nn.ModuleList(
            [
                TPSPTransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    attn_bias=attn_bias,
                    ffn_bias=ffn_bias,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = ColumnParallelLinear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        S_local = S // self.ws

        # Embedding (same on all GPUs, full sequence)
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)  # (B, S, h)

        # Scatter to SP region: each GPU takes its chunk of the sequence
        start = self.rank * S_local
        x = x[:, start : start + S_local, :].contiguous()  # (B, S/N, h)
        x = self.emb_dropout(x)

        # All blocks stay in SP region
        for block in self.blocks:
            x = block(x)

        # Gather back for output
        x = self.norm_f(x)  # (B, S/N, h) -- SP region
        x = _AllGatherFromSPRegion.apply(x)  # (B, S, h) -- full sequence
        return self.lm_head(x)  # (B, S, vocab/N)


# ================================================================
#  Differentiable all-gather for TP cross entropy
# ================================================================


class _AllGatherFromParallelRegion(torch.autograd.Function):
    """All-gather in forward, chunk/scatter in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ws = dist.get_world_size()
        ctx.ws = ws
        gathered = [torch.zeros_like(x) for _ in range(ws)]
        dist.all_gather(gathered, x.contiguous())
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        rank = dist.get_rank()
        chunks = grad.chunk(ctx.ws, dim=-1)
        return chunks[rank].contiguous()


def tp_cross_entropy(logits_local, labels, vocab_size):
    full = _AllGatherFromParallelRegion.apply(logits_local)
    return F.cross_entropy(full.view(-1, vocab_size), labels.view(-1))


# ================================================================
#  Main
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(description="TP+SP GPT benchmark")
    p.add_argument("--d-model", type=int, default=D_MODEL)
    p.add_argument("--n-heads", type=int, default=N_HEADS)
    p.add_argument("--d-ff", type=int, default=D_FF)
    p.add_argument("--n-layers", type=int, default=N_LAYERS)
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--attn-bias", action="store_true", default=False)
    p.add_argument("--no-ffn-bias", action="store_true", default=False)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--warmup", type=int, default=NUM_WARMUP)
    p.add_argument("--benchmark", type=int, default=NUM_BENCHMARK)
    p.add_argument("--output-dir", type=str, default="outputs")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
    )

    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank = dist.get_rank()
    ws = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    if rank == 0:
        logger.info("TP+SP benchmark: %d GPUs", ws)
        logger.info(
            "d_model=%d, n_heads=%d, d_ff=%d, n_layers=%d, vocab=%d",
            args.d_model,
            args.n_heads,
            args.d_ff,
            args.n_layers,
            args.vocab_size,
        )
        logger.info(
            "Each GPU: %d/%d heads, %d/%d FFN dim, seq_chunk=%d/%d",
            args.n_heads // ws,
            args.n_heads,
            args.d_ff // ws,
            args.d_ff,
            args.seq_len // ws,
            args.seq_len,
        )

    model = TPSPGPT(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        attn_bias=args.attn_bias,
        ffn_bias=not args.no_ffn_bias,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mem_model = get_gpu_memory_mb(device)
    n_local = count_parameters(model)

    if rank == 0:
        logger.info(
            "Model params: %s (local), %s (total)", f"{n_local:,}", f"{n_local*ws:,}"
        )
        logger.info("Model size: %.2f MB", mem_model)

    input_ids = torch.randint(
        0, args.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    labels = torch.randint(
        0, args.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    for _ in range(args.warmup):
        logits = model(input_ids)
        loss = tp_cross_entropy(logits, labels, args.vocab_size)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    dist.barrier()

    fwd_t, bwd_t, step_t = [], [], []
    for _ in range(args.benchmark):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        logits = model(input_ids)
        loss = tp_cross_entropy(logits, labels, args.vocab_size)
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        dist.barrier()
        t2 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        dist.barrier()
        t3 = time.perf_counter()
        fwd_t.append(t1 - t0)
        bwd_t.append(t2 - t1)
        step_t.append(t3 - t0)

    peak = get_gpu_peak_memory_mb(device)

    for r in range(ws):
        if rank == r:
            logger.info(
                "[GPU %d] params: %s | model: %.1f MB | peak: %.1f MB",
                rank,
                f"{n_local:,}",
                mem_model,
                peak,
            )
        dist.barrier()

    if rank == 0:
        average = lambda values: sum(values) / len(values)

        results = {
            "mode": f"tp_sp_{ws}",
            "world_size": ws,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "n_layers": args.n_layers,
            "vocab_size": args.vocab_size,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "params_per_gpu": n_local,
            "mem_model_mb": round(mem_model, 2),
            "mem_peak_mb": round(peak, 2),
            "fwd_ms": round(average(fwd_t) * 1000, 3),
            "bwd_ms": round(average(bwd_t) * 1000, 3),
            "step_ms": round(average(step_t) * 1000, 3),
            "tokens_per_sec": round(
                args.batch_size * args.seq_len / average(step_t), 1
            ),
            "throughput": round(BATCH_SIZE * SEQ_LEN / average(step_t), 1),
            "loss": round(loss.item(), 4),
        }

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "results_model_gpt_tp_sp.json")
        with open(out_path, "w") as fout:
            json.dump(results, fout, indent=2)

        logger.info("=" * 60)
        logger.info(
            "  Model GPT - Tensor Parallelism + Sequence Parallelism - %d GPUs", ws
        )
        logger.info("=" * 60)

        logger.info(f"  Forward:     {results['fwd_ms']:>8.2f} ms")
        logger.info(f"  Backward:    {results['bwd_ms']:>8.2f} ms")
        logger.info(f"  Full step:   {results['step_ms']:>8.2f} ms")
        logger.info(f"  Throughput:  {results['throughput']:>8.0f} tok/s")
        logger.info(f"  Peak memory: {results['mem_peak_mb']:>8.1f} MB")
        logger.info("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
