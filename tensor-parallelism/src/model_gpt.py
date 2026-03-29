"""Standard (non-parallelized) GPT-style transformer model.

A minimal GPT decoder used as the baseline for tensor-parallelism experiments.
Single-GPU benchmark - no distributed communication.
"""

import argparse
import json
import logging
import math
import time

from rich.logging import RichHandler

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_parameters, get_gpu_memory_mb, get_gpu_peak_memory_mb

logger = logging.getLogger(__name__)

# ================================================================
# Default Model & Benchmark Constants
# ================================================================

DEFAULT_D_MODEL = 512       # hidden dimension (embedding size)
DEFAULT_N_HEADS = 8         # number of attention heads
DEFAULT_D_FF = 2048         # feed-forward intermediate dimension (4x D_MODEL)
DEFAULT_N_LAYERS = 6        # number of transformer blocks
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_MAX_SEQ_LEN = 512

DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 256
DEFAULT_NUM_WARMUP = 3
DEFAULT_NUM_BENCHMARK = 10


# ================================================================
# Standard Multi-Head Attention
# ================================================================


class StandardAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


# ================================================================
# Standard FFN (feed-forward network)
# ================================================================


class StandardFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=bias)
        self.W2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.gelu(self.W1(x)))


# ================================================================
# GPT Transformer Block
# ================================================================


class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 attn_bias: bool = False, ffn_bias: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardAttention(d_model, n_heads, bias=attn_bias)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = StandardFFN(d_model, d_ff, bias=ffn_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ================================================================
# Full GPT Model
# ================================================================


class StandardGPT(nn.Module):
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
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads, d_ff,
                                    attn_bias=attn_bias, ffn_bias=ffn_bias)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# ================================================================
#  Main
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Standard GPT benchmark (single GPU)")
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

    device = torch.device("cuda")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    logger.info("Standard GPT benchmark (single GPU)")
    logger.info(
        "d_model=%d, n_heads=%d, d_ff=%d, n_layers=%d, vocab=%d",
        args.d_model, args.n_heads, args.d_ff, args.n_layers, args.vocab_size,
    )

    model = StandardGPT(
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
    n_params = count_parameters(model)

    logger.info("Model params: %s", f"{n_params:,}")
    logger.info("Model size: %.2f MB", mem_model)

    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    for _ in range(args.warmup):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    fwd_t, bwd_t, step_t = [], [], []
    for _ in range(args.benchmark):
        optimizer.zero_grad()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), labels.view(-1))
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        fwd_t.append(t1 - t0)
        bwd_t.append(t2 - t1)
        step_t.append(t3 - t0)

    peak = get_gpu_peak_memory_mb(device)

    logger.info(
        "params: %s | model: %.1f MB | peak: %.1f MB",
        f"{n_params:,}", mem_model, peak,
    )

    average = lambda values: sum(values) / len(values)
    results = dict(
        mode="no_tp",
        num_gpus=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        params=n_params,
        mem_model_mb=round(mem_model, 2),
        mem_peak_mb=round(peak, 2),
        fwd_ms=round(average(fwd_t) * 1000, 3),
        bwd_ms=round(average(bwd_t) * 1000, 3),
        step_ms=round(average(step_t) * 1000, 3),
        tokens_per_sec=round(args.batch_size * args.seq_len / average(step_t), 1),
        loss=round(loss.item(), 4),
    )
    with open("results_model_gpt.json", "w") as fout:
        json.dump(results, fout, indent=2)
    logger.info("=" * 60)
    logger.info("  Model GPT - No parallelism - 1 GPU")
    logger.info("=" * 60)
    for k in [
        "params",
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


if __name__ == "__main__":
    main()
