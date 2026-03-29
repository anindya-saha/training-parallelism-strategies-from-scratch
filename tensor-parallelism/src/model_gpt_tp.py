"""TP GPT-style transformer model.

A minimal GPT decoder with Tensor Parallelism. Weight matrices are sharded
across GPUs using Column-Parallel and Row-Parallel linear layers.

Uses the default process group (all GPUs) for all communication.
"""

import argparse
import json
import logging
import math
import os
import time

from rich.logging import RichHandler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils import count_parameters, get_gpu_memory_mb, get_gpu_peak_memory_mb

logger = logging.getLogger(__name__)

# ================================================================
# Default Model & Benchmark Constants
# ================================================================

DEFAULT_D_MODEL = 512  # hidden dimension (embedding size)
DEFAULT_N_HEADS = 8  # number of attention heads
DEFAULT_D_FF = 2048  # feed-forward intermediate dimension (4x D_MODEL)
DEFAULT_N_LAYERS = 6  # number of transformer blocks
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_MAX_SEQ_LEN = 512

DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 256
DEFAULT_NUM_WARMUP = 3
DEFAULT_NUM_BENCHMARK = 10


# ================================================================
#  Autograd functions for TP communication
# ================================================================
# _CopyToParallelRegion:     identity forward,   all-reduce backward
#   -> Used before column-parallel layers
# _ReduceFromParallelRegion: all-reduce forward, identity backward
#   -> Used after row-parallel layers


class _CopyToParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


class _ReduceFromParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad


# ================================================================
#  Column-parallel linear
# ================================================================
# Full: (d_out, d_in) -> this GPU: (d_out // N, d_in)
# Input: same on all GPUs -> output: each GPU gets different slice
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
        x = _CopyToParallelRegion.apply(x)
        return F.linear(x, self.weight, self.bias)


# ================================================================
#  Row-parallel linear
# ================================================================
# Full: (d_out, d_in) -> this GPU: (d_out, d_in // N)
# Input: local slice -> output: partial sum, needs all-reduce!


class RowParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        ws = dist.get_world_size()
        assert d_in % ws == 0, "d_in must be divisible by world size"
        # F.linear computes x @ W.T, so W is stored as (d_out, d_in) to match PyTorch convention.
        # We shard the input dim: each GPU holds (d_out, d_in // ws).
        self.weight = nn.Parameter(torch.empty(d_out, d_in // ws))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Bias only on rank 0: _ReduceFromParallelRegion all-reduces the output,
        # so if every rank added bias, it would be summed ws times.
        self.bias = None
        if bias and dist.get_rank() == 0:
            self.bias = nn.Parameter(torch.empty(d_out))
            nn.init.uniform_(self.bias, a=-1 / math.sqrt(d_in), b=1 / math.sqrt(d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        return _ReduceFromParallelRegion.apply(out)


# ================================================================
# TP attention (split by heads)
# ================================================================


class TPAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads_local, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads_local, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads_local, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


# ================================================================
# TP FFN (feed-forward network)
# ================================================================


class TPFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.W1 = ColumnParallelLinear(d_model, d_ff, bias=bias)
        self.W2 = RowParallelLinear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.gelu(self.W1(x)))


# ================================================================
# TP GPT transformer block
# ================================================================


class TPGPTTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_bias: bool = False,
        ffn_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TPAttention(d_model, n_heads, bias=attn_bias)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = TPFFN(d_model, d_ff, bias=ffn_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ================================================================
# Full TP GPT model
# ================================================================


class TPGPT(nn.Module):
    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        d_ff: int = DEFAULT_D_FF,
        n_layers: int = DEFAULT_N_LAYERS,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        attn_bias: bool = False,
        ffn_bias: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TPGPTTransformerBlock(
                    d_model, n_heads,
                    d_ff,
                    attn_bias=attn_bias,
                    ffn_bias=ffn_bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = ColumnParallelLinear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


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
    p = argparse.ArgumentParser(description="Tensor parallelism GPT benchmark")
    p.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    p.add_argument("--d-ff", type=int, default=DEFAULT_D_FF)
    p.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--attn-bias", action="store_true", default=False)
    p.add_argument("--no-ffn-bias", action="store_true", default=False)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--warmup", type=int, default=DEFAULT_NUM_WARMUP)
    p.add_argument("--benchmark", type=int, default=DEFAULT_NUM_BENCHMARK)
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)],
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
        logger.info("TP benchmark: %d GPUs", ws)
        logger.info(
            "d_model=%d, n_heads=%d, d_ff=%d, n_layers=%d, vocab=%d",
            args.d_model,
            args.n_heads,
            args.d_ff,
            args.n_layers,
            args.vocab_size,
        )
        logger.info(
            "Each GPU: %d/%d heads, %d/%d FFN dim",
            args.n_heads // ws,
            args.n_heads,
            args.d_ff // ws,
            args.d_ff,
        )

    model = TPGPT(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        attn_bias=args.attn_bias,
        ffn_bias=not args.no_ffn_bias,
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
        results = dict(
            mode=f"tp_{ws}",
            num_gpus=ws,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            params_per_gpu=n_local,
            mem_model_mb=round(mem_model, 2),
            mem_peak_mb=round(peak, 2),
            fwd_ms=round(average(fwd_t) * 1000, 3),
            bwd_ms=round(average(bwd_t) * 1000, 3),
            step_ms=round(average(step_t) * 1000, 3),
            tokens_per_sec=round(args.batch_size * args.seq_len / average(step_t), 1),
            loss=round(loss.item(), 4),
        )
        with open("results_model_gpt_tp.json", "w") as fout:
            json.dump(results, fout, indent=2)
        logger.info("=" * 60)
        logger.info("  Model GPT - Tensor parallelism - %d GPUs", ws)
        logger.info("=" * 60)
        for k in [
            "params_per_gpu",
            "mem_model_mb",
            "mem_peak_mb",
            "fwd_ms",
            "bwd_ms",
            "step_ms",
            "tokens_per_sec",
            "loss",
        ]:
            v = results[k]
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                logger.info("  %-25s %12.2f", label, v)
            else:
                logger.info("  %-25s %12s", label, f"{v:,}")
        logger.info("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
