"""Standard (non-parallelized) Llama-style transformer model.

A minimal Llama decoder used as the baseline for tensor-parallelism experiments.
Differences from the GPT model (model_gpt.py):
  1. RMSNorm instead of LayerNorm
  2. Rotary Position Embeddings (RoPE) instead of learned positional embeddings
  3. Grouped Query Attention (GQA) instead of standard multi-head attention
  4. SwiGLU FFN instead of GELU FFN (3 weight matrices instead of 2)
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
# Default model & benchmark constants
# ================================================================

DEFAULT_D_MODEL = 512       # hidden dimension (embedding size)
DEFAULT_N_HEADS = 8         # number of Q attention heads
DEFAULT_N_KV_HEADS = 4      # number of KV heads (GQA)
DEFAULT_N_LAYERS = 6        # number of transformer blocks
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_MAX_SEQ_LEN = 512

# SwiGLU uses 2/3 of the typical 4x expansion to keep param count similar,
# since it has 3 matrices instead of 2.  floor to multiple of 256.
DEFAULT_D_FF = (int(DEFAULT_D_MODEL * 8 / 3) // 256) * 256  # = 1024

DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 256
DEFAULT_NUM_WARMUP = 3
DEFAULT_NUM_BENCHMARK = 10


# ================================================================
# RMSNorm (replaces LayerNorm)
# ================================================================


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ================================================================
# Rotary Position Embeddings (RoPE)
# ================================================================


def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x_complex.shape[-2], :].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


# ================================================================
# Grouped Query Attention (replaces standard MHA)
# ================================================================


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int, bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=bias)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.W_o = nn.Linear(n_heads * self.d_head, d_model, bias=bias)

        # Registered as a buffer (not a parameter) so it moves with the model
        # to the correct device but is not updated by the optimizer.
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.d_head, max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        Q = apply_rope(Q, self.rope_freqs)
        K = apply_rope(K, self.rope_freqs)

        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


# ================================================================
# SwiGLU FFN (replaces GELU FFN)
# ================================================================


class SwiGLUFFN(nn.Module):
    """output = W_down( SiLU(W_gate(x)) * W_up(x) )"""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.W_up = nn.Linear(d_model, d_ff, bias=bias)
        self.W_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


# ================================================================
# Llama transformer block
# ================================================================


class LlamaTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, max_seq_len: int,
                 attn_bias: bool = False, ffn_bias: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len,
                                         bias=attn_bias)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, bias=ffn_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ================================================================
# Full Llama model
# ================================================================


class StandardLlama(nn.Module):
    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        n_kv_heads: int = DEFAULT_N_KV_HEADS,
        d_ff: int = DEFAULT_D_FF,
        n_layers: int = DEFAULT_N_LAYERS,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        attn_bias: bool = False,
        ffn_bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            LlamaTransformerBlock(d_model, n_heads, n_kv_heads, d_ff, max_seq_len,
                                  attn_bias=attn_bias, ffn_bias=ffn_bias)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


# ================================================================
#  Main
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Standard Llama benchmark (single GPU)")
    p.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    p.add_argument("--n-kv-heads", type=int, default=DEFAULT_N_KV_HEADS)
    p.add_argument("--d-ff", type=int, default=DEFAULT_D_FF)
    p.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--attn-bias", action="store_true", default=False)
    p.add_argument("--ffn-bias", action="store_true", default=False)
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

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    logger.info("Standard Llama benchmark (single GPU)")
    logger.info(
        "d_model=%d, n_heads=%d, n_kv_heads=%d, d_ff=%d, n_layers=%d, vocab=%d",
        args.d_model, args.n_heads, args.n_kv_heads,
        args.d_ff, args.n_layers, args.vocab_size,
    )

    model = StandardLlama(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        attn_bias=args.attn_bias,
        ffn_bias=args.ffn_bias,
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
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), labels.view(-1))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
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
        model="llama",
        num_gpus=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        params_per_gpu=n_params,
        mem_model_mb=round(mem_model, 2),
        mem_peak_mb=round(peak, 2),
        fwd_ms=round(average(fwd_t) * 1000, 3),
        bwd_ms=round(average(bwd_t) * 1000, 3),
        step_ms=round(average(step_t) * 1000, 3),
        tokens_per_sec=round(args.batch_size * args.seq_len / average(step_t), 1),
        loss=round(loss.item(), 4),
    )
    with open("results_model_llama.json", "w") as fout:
        json.dump(results, fout, indent=2)
    logger.info("=" * 60)
    logger.info("  Model Llama - No parallelism - 1 GPU")
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


if __name__ == "__main__":
    main()
