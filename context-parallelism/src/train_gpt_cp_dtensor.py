"""GPT-2 with Context Parallelism via PyTorch's context_parallel() API.

Same GPT-2 architecture as train_gpt.py, but uses F.scaled_dot_product_attention
(SDPA) instead of explicit Q @ K^T.  Context Parallelism is applied via a single
context manager - no model modifications, no InnerAttention wrapper, no
parallelize_module, no private APIs.

    with context_parallel(cp_mesh, buffers=(...), buffer_seq_dims=(...)):
        logits = model(input_ids, position_ids)
        loss = F.cross_entropy(...)
        loss.backward()

The context_parallel() context manager (public API since PyTorch 2.7) does two
things:
  1. Shards the buffers in-place along the specified sequence dimensions.
  2. Replaces F.scaled_dot_product_attention with Ring Attention for all
     calls within the block.

No InnerAttention wrapper is needed because the context manager intercepts
SDPA globally - it does not use parallelize_module hooks that require a
specific module boundary.

Contrast with train_gpt_cp.py (hand-written ring attention):
  - No _ring_rotate, no _ring_attention, no P2P ops in the model.
  - No cp_group threaded through __init__.
  - Uses DeviceMesh instead of raw dist.new_group.
  - The model is byte-for-byte identical to the single-GPU baseline
    (train_gpt.py) except it uses SDPA instead of explicit Q @ K^T.

Reference:
  https://docs.pytorch.org/tutorials/unstable/context_parallel.html

Example:
    torchrun --nproc_per_node=2 src/train_gpt_cp_dtensor.py --config mini --cp-size 2
    torchrun --nproc_per_node=4 src/train_gpt_cp_dtensor.py --config medium --seq-len 1024 --cp-size 4
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import _cp_options, set_rotate_method
from torch.nn.attention import SDPBackend, sdpa_kernel
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
# Multi-Head Attention (plain SDPA - no parallelism code)
# ================================================================


class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.dropout = config.dropout

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d_head]
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d_head]
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d_head]

        p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=p)  # [B, H, T, d_head]

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, d_model]
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
        x = x + self.attn(self.ln1(x))  # [B, T, d_model]
        x = x + self.ffn(self.ln2(x))  # [B, T, d_model]
        return x


# ================================================================
# Full GPT Model (plain nn.Module - no parallelism code)
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

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape  # [B, T_local] after CP shard
        x = self.emb_dropout(self.tok_emb(input_ids) + self.pos_emb(position_ids))  # [B, T_local, d_model]
        for block in self.blocks:
            x = block(x)  # [B, T_local, d_model]
        x = self.ln_f(x)  # [B, T_local, d_model]
        return self.lm_head(x)  # [B, T_local, vocab_size]


# ================================================================
#  Main
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(description="GPT-2 benchmark (DTensor CP)")
    p.add_argument(
        "--config",
        type=str,
        default="mini",
        choices=list(GPT_CONFIGS.keys()),
        help="Model configuration: mini (~19M), small (~117M), medium (~345M)",
    )
    p.add_argument("--cp-size", type=int, default=1, help="Context parallelism size")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--warmup", type=int, default=DEFAULT_NUM_WARMUP)
    p.add_argument("--benchmark", type=int, default=DEFAULT_NUM_BENCHMARK)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Model dtype (use bfloat16 for apples-to-apples comparison with DTensor CP)",
    )
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

    assert args.cp_size <= world_size, f"cp_size={args.cp_size} > world_size={world_size}"
    assert args.cp_size > 0, f"cp_size must be > 0, got {args.cp_size}"

    # --- Device setup ---
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    config = GPT_CONFIGS[args.config]
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # --- DeviceMesh for CP ---
    cp_mesh = init_device_mesh("cuda", (args.cp_size,), mesh_dim_names=("cp",))

    if rank == 0:
        logger.info(
            "GPT-2 benchmark (DTensor CP) - config: %s, world_size: %d, cp_size: %d",
            args.config,
            world_size,
            args.cp_size,
        )
        logger.info(
            "d_model=%d, n_heads=%d, d_ff=%d, n_layers=%d, vocab=%d, cp_size=%d",
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.n_layers,
            config.vocab_size,
            args.cp_size,
        )

    # --- Build model (plain nn.Module, no CP code) ---
    # Flash Attention requires bfloat16 or float16 inputs.
    model = GPT(config).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    mem_model = get_gpu_memory_mb(device)
    n_params = count_parameters(model)

    if rank == 0:
        logger.info("Model params: %s", f"{n_params:,}")
        logger.info("Model size: %.2f MB", mem_model)

    # --- Prepare full-sequence inputs ---
    seq_len = min(args.seq_len, config.max_seq_len)
    assert seq_len % (2 * args.cp_size) == 0, (
        f"seq_len must be divisible by 2*cp_size for _HeadTailLoadBalancer: "
        f"seq_len={seq_len} % (2*{args.cp_size}) != 0"
    )

    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    labels = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)

    # --- context_parallel: shard inputs + replace SDPA with Ring Attention ---
    #
    # The context manager does two things:
    #   1. Shards (input_ids, position_ids, labels) in-place along dim 1
    #      (the sequence dimension).  After this, each GPU holds T/CP tokens.
    #   2. Replaces F.scaled_dot_product_attention with Ring Attention for
    #      all calls within the block.  No model changes needed.
    #
    # _cp_options.enable_load_balance enables head-tail (zig-zag) reordering
    # before sharding.  Contrast with train_gpt_cp.py which uses contiguous
    # chunks (no load balancing): GPU 0 gets tokens [0..T/CP-1], GPU 1 gets
    # [T/CP..2*T/CP-1], etc.  With load balancing, each GPU receives a mix
    # of head (cheap) and tail (expensive) causal positions, e.g.
    # [0, 7, 1, 6, 2, 5, 3, 4] for S=8, CP=2.  This is True by default
    # but we set it explicitly for clarity.
    #
    # set_rotate_method selects the KV shard rotation strategy:
    #   "alltoall" - interleaved all-to-all collectives to overlap SDPA
    #                computation with the communication for the next step.
    #   "allgather" (default) - all-gather based pass-KV (used in Llama3).
    #
    # Reference: https://docs.pytorch.org/tutorials/unstable/context_parallel.html
    _cp_options.enable_load_balance = True
    set_rotate_method("alltoall")

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with context_parallel(
            cp_mesh,
            buffers=(input_ids, position_ids, labels),
            buffer_seq_dims=(1, 1, 1),
        ):
            # --- Warmup ---
            if rank == 0:
                logger.info("--- Warmup (%d steps) ---", args.warmup)
            for step in range(args.warmup):
                logits = model(input_ids, position_ids)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
                loss_avg = loss.detach().clone()
                dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if rank == 0:
                    logger.info(
                        "  warmup %d/%d  loss=%.4f",
                        step + 1,
                        args.warmup,
                        loss_avg.item(),
                    )

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

            # --- Benchmark ---
            if rank == 0:
                logger.info("--- Benchmark (%d steps) ---", args.benchmark)
            fwd_t, bwd_t, step_t = [], [], []
            for step in range(args.benchmark):
                optimizer.zero_grad()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                logits = model(input_ids, position_ids)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
                loss_avg = loss.detach().clone()
                dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG)
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
                        loss_avg.item(),
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
        mode="cp_dtensor",
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
        cp_size=args.cp_size,
        params_per_gpu=n_params,
        mem_model_mb=round(mem_model, 2),
        mem_peak_mb=round(peak, 2),
        fwd_ms=round(average(fwd_t) * 1000, 3),
        bwd_ms=round(average(bwd_t) * 1000, 3),
        step_ms=round(average(step_t) * 1000, 3),
        tokens_per_sec=round(args.batch_size * seq_len / average(step_t), 1),
        loss=round(loss_avg.item(), 4),
    )

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"results_train_gpt_cp_dtensor_{args.config}.json")
        with open(out_path, "w") as fout:
            json.dump(results, fout, indent=2)
        logger.info("=" * 60)
        logger.info(
            "  GPT-2 %s - Context Parallelism DTensor (CP=%d) - %d GPU(s)",
            args.config.upper(),
            args.cp_size,
            world_size,
        )
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
