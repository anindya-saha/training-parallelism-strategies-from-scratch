"""GPT-2 style transformer model -- baseline for all parallelism experiments.

Follows the GPT-2 architecture:
- Pre-norm (LayerNorm before attention/FFN)
- Dropout in 3 locations: after embeddings, after attention softmax, after
  residual projections (attention output and FFN output)
- Causal mask registered as a buffer (not recreated every forward call)
- Explicit Q @ K^T attention (not SDPA) so each operation is visible

Three preset configurations:
    --config mini     ~19M params  (fast local experiments)
    --config small    ~117M params (GPT-2 Small)
    --config medium   ~345M params (GPT-2 Medium)

Distributed-ready: always initializes a process group so it can be launched
with torchrun for multi-GPU runs. Each GPU runs the same model independently
(no parallelism in this baseline -- that comes in the CP/TP/SP variants).

Example:
    torchrun --nproc_per_node=1 src/train_gpt.py
    torchrun --nproc_per_node=1 src/train_gpt.py --config small
    torchrun --nproc_per_node=2 src/train_gpt.py --config medium --seq-len 512
"""

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from utils import count_parameters, get_gpu_memory_mb, get_gpu_peak_memory_mb

logger = logging.getLogger(__name__)


# ================================================================
# Model Configurations
# ================================================================


@dataclass
class GPTConfig:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 6
    vocab_size: int = 10_000
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = True


GPT_CONFIGS = {
    "mini": GPTConfig(
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        vocab_size=10_000,
        max_seq_len=512,
        dropout=0.1,
        bias=True,
    ),
    "small": GPTConfig(
        d_model=768,
        n_heads=12,
        d_ff=3072,
        n_layers=12,
        vocab_size=50_257,
        max_seq_len=1024,
        dropout=0.1,
        bias=True,
    ),
    "medium": GPTConfig(
        d_model=1024,
        n_heads=16,
        d_ff=4096,
        n_layers=24,
        vocab_size=50_257,
        max_seq_len=1024,
        dropout=0.1,
        bias=True,
    ),
}

DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 256
DEFAULT_NUM_WARMUP = 3
DEFAULT_NUM_BENCHMARK = 10

# ================================================================
# Multi-Head Attention
# ================================================================


class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Register the causal mask as a buffer rather than creating it in
        # forward(). Buffers are:
        #  - allocated once, not every forward call (avoids T*T allocation)
        #  - moved with the model on .to(device) / .cuda()
        #  - included in state_dict for checkpointing
        #  - NOT treated as parameters (no gradients)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        scale = 1.0 / math.sqrt(self.d_head)

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.W_o(out))


# ================================================================
# FFN (feed-forward network)
# ================================================================


class FFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.W2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resid_dropout(self.W2(F.gelu(self.W1(x))))


# ================================================================
# Transformer Block
# ================================================================


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ================================================================
# Full GPT Model
# ================================================================


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.emb_dropout(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# ================================================================
#  Main
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(description="GPT-2 benchmark")
    p.add_argument(
        "--config",
        type=str,
        default="mini",
        choices=list(GPT_CONFIGS.keys()),
        help="Model configuration: mini (~19M), small (~117M), medium (~345M)",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--warmup", type=int, default=DEFAULT_NUM_WARMUP)
    p.add_argument("--benchmark", type=int, default=DEFAULT_NUM_BENCHMARK)
    p.add_argument("--output-dir", type=str, default="outputs")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
    )

    args = parse_args()

    # --- Distributed setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    config = GPT_CONFIGS[args.config]

    if rank == 0:
        logger.info("GPT-2 benchmark -- config: %s, world_size: %d", args.config, world_size)
        logger.info(
            "d_model=%d, n_heads=%d, d_ff=%d, n_layers=%d, vocab=%d, dropout=%.2f",
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.n_layers,
            config.vocab_size,
            config.dropout,
        )

    model = GPT(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    mem_model = get_gpu_memory_mb(device)
    n_params = count_parameters(model)

    if rank == 0:
        logger.info("Model params: %s", f"{n_params:,}")
        logger.info("Model size: %.2f MB", mem_model)

    seq_len = min(args.seq_len, config.max_seq_len)
    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)

    for step in range(args.warmup):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    if rank == 0:
        logger.info("--- Benchmark (%d steps) ---", args.benchmark)
    fwd_t, bwd_t, step_t = [], [], []
    for step in range(args.benchmark):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
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
        if rank == 0:
            logger.info(
                "  step %d/%d  loss=%.4f  fwd=%.1fms  bwd=%.1fms  total=%.1fms",
                step + 1,
                args.benchmark,
                loss.item(),
                (t1 - t0) * 1000,
                (t2 - t1) * 1000,
                (t3 - t0) * 1000,
            )

    peak = get_gpu_peak_memory_mb(device)

    if rank == 0:
        logger.info(
            "params: %s | model: %.1f MB | peak: %.1f MB",
            f"{n_params:,}",
            mem_model,
            peak,
        )

    average = lambda values: sum(values) / len(values)
    results = dict(
        mode="baseline",
        config=args.config,
        rank=rank,
        num_gpus=world_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
        batch_size=args.batch_size,
        seq_len=seq_len,
        params_per_gpu=n_params,
        mem_model_mb=round(mem_model, 2),
        mem_peak_mb=round(peak, 2),
        fwd_ms=round(average(fwd_t) * 1000, 3),
        bwd_ms=round(average(bwd_t) * 1000, 3),
        step_ms=round(average(step_t) * 1000, 3),
        tokens_per_sec=round(args.batch_size * seq_len / average(step_t), 1),
        loss=round(loss.item(), 4),
    )

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"results_train_gpt_{args.config}.json")
        with open(out_path, "w") as fout:
            json.dump(results, fout, indent=2)
        logger.info("=" * 60)
        logger.info("  GPT-2 %s - No parallelism - %d GPU(s)", args.config.upper(), world_size)
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
