"""GPT-2 with hand-written Context Parallelism (Ring Attention).

Same GPT-2 architecture as train_gpt.py (pre-norm, dropout, explicit Q@K^T),
but attention is computed via ring attention across the CP group:
  1. The input sequence is split into chunks, one per CP rank.
  2. Each rank computes Q, K, V from its local chunk.
  3. Q stays local; K and V rotate around the ring for cp_size steps.
  4. At each step, partial attention is merged via online softmax (m, l, o).
  5. After all steps, each rank has the correct attention output for its chunk.
  6. Loss is all-reduced across CP ranks (each rank only sees its chunk's labels).

Everything outside attention (embeddings, FFN, LayerNorm, lm_head) runs
locally on each rank's chunk with no communication.

Example:
    torchrun --nproc_per_node=2 src/train_gpt_cp.py --config mini --cp-size 2
    torchrun --nproc_per_node=4 src/train_gpt_cp.py --config medium --seq-len 1024 --cp-size 4
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
# Multi-Head Attention with Ring Attention (Context Parallelism)
# ================================================================


class Attention(nn.Module):
    def __init__(self, config: GPTConfig, cp_group):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.cp_group = cp_group

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _ring_rotate(self, k, v):
        """Rotate KV one step around the ring: send to next, receive from previous.

        Args:
            k: [B, H, T_local, d_head]
            v: [B, H, T_local, d_head]
        Returns:
            k_new, v_new: [B, H, T_local, d_head] received from previous rank
        """
        cp_rank = dist.get_rank(self.cp_group)
        cp_size = dist.get_world_size(self.cp_group)

        next_rank = (cp_rank + 1) % cp_size
        prev_rank = (cp_rank - 1 + cp_size) % cp_size

        global_ranks = dist.get_process_group_ranks(self.cp_group)
        next_global = global_ranks[next_rank]
        prev_global = global_ranks[prev_rank]

        k_new = torch.empty(k.shape, dtype=k.dtype, device=k.device)  # [B, H, T_local, d_head]
        v_new = torch.empty(v.shape, dtype=v.dtype, device=v.device)  # [B, H, T_local, d_head]

        p2p_ops = [
            dist.P2POp(dist.isend, k.contiguous(), next_global),
            dist.P2POp(dist.irecv, k_new, prev_global),
            dist.P2POp(dist.isend, v.contiguous(), next_global),
            dist.P2POp(dist.irecv, v_new, prev_global),
        ]
        reqs = dist.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

        return k_new, v_new

    def _ring_attention(self, q_local, k_local, v_local):
        """Compute causal self-attention via ring attention across CP ranks.

        Each rank holds Q for its local chunk and rotates K,V around the ring,
        accumulating partial attention results with online softmax.

        Args:
            q_local: [B, H, T_local, d_head] - queries for this rank's chunk
            k_local: [B, H, T_local, d_head] - keys for this rank's chunk
            v_local: [B, H, T_local, d_head] - values for this rank's chunk
        Returns:
            output: [B, H, T_local, d_head]
        """
        cp_rank = dist.get_rank(self.cp_group)
        cp_size = dist.get_world_size(self.cp_group)

        B, H, T_local, d_head = q_local.shape
        scale = d_head**-0.5

        # KV buffers that rotate around the ring
        k_recv = k_local.clone()  # [B, H, T_local, d_head]
        v_recv = v_local.clone()  # [B, H, T_local, d_head]

        # Online softmax accumulators (fp32 for numerical stability)
        # Initialize accumulators for online softmax
        # o_acc: running weighted sum (unnormalized)
        # m: running max per query position
        # l: running sum of exponentials per query position
        o_acc = torch.zeros(B, H, T_local, d_head, device=q_local.device, dtype=torch.float32)
        m = torch.full((B, H, T_local, 1), float("-inf"), device=q_local.device, dtype=torch.float32)
        l = torch.zeros(B, H, T_local, 1, device=q_local.device, dtype=torch.float32)

        for step in range(cp_size):
            # Which rank's KV are we currently holding?
            source_rank = (cp_rank - step + cp_size) % cp_size

            # Attention scores for this tile: Q_local @ K_source^T
            scores = (q_local @ k_recv.transpose(-2, -1)) * scale  # [B, H, T_local, T_local]

            # Causal mask depends on the relative position of source vs local chunk
            if source_rank == cp_rank:
                # Diagonal tile: same chunk -> standard causal mask within the chunk
                causal_mask = torch.triu(
                    torch.ones(T_local, T_local, device=q_local.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            elif source_rank > cp_rank:
                # Future tokens: mask everything (no query should attend to future KV)
                scores.fill_(float("-inf"))
            # else: source_rank < cp_rank -> past tokens, attend fully (no mask)

            # Online softmax: merge this tile into running accumulators (m, l, o_acc)
            block_max = scores.max(dim=-1, keepdim=True).values  # [B, H, T_local, 1]
            # Clamp -inf maxes to avoid nan in exp
            block_max = block_max.clamp(min=-1e30)

            block_exp = torch.exp(scores - block_max)  # [B, H, T_local, T_local]
            block_sum = block_exp.sum(dim=-1, keepdim=True)  # [B, H, T_local, 1]
            block_out = block_exp @ v_recv  # [B, H, T_local, d_head]

            # Rescale old and new contributions to the new global max
            m_new = torch.maximum(m, block_max)  # [B, H, T_local, 1]
            exp_old = torch.exp(m - m_new)  # [B, H, T_local, 1]
            exp_new = torch.exp(block_max - m_new)  # [B, H, T_local, 1]

            l = exp_old * l + exp_new * block_sum  # [B, H, T_local, 1]
            o_acc = exp_old * o_acc + exp_new * block_out  # [B, H, T_local, d_head]
            m = m_new

            # Rotate KV to the next rank (skip on last step)
            if step < cp_size - 1:
                k_recv, v_recv = self._ring_rotate(k_recv, v_recv)

        # Normalize by the accumulated denominator
        output = o_acc / l.clamp(min=1e-8)  # [B, H, T_local, d_head]
        return self.attn_dropout(output.to(q_local.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # [B, T_local, d_model]

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T_local, d_head]
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T_local, d_head]
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T_local, d_head]

        out = self._ring_attention(Q, K, V)  # [B, H, T_local, d_head]

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T_local, d_model]
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
    def __init__(self, config: GPTConfig, cp_group):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config, cp_group)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # [B, T_local, d_model]
        x = x + self.ffn(self.ln2(x))  # [B, T_local, d_model]
        return x


# ================================================================
# Full GPT Model
# ================================================================


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, cp_group):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config, cp_group) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape  # [B, T_local] after CP split
        x = self.emb_dropout(self.tok_emb(input_ids) + self.pos_emb(position_ids))  # [B, T_local, d_model]
        for block in self.blocks:
            x = block(x)  # [B, T_local, d_model]
        x = self.ln_f(x)  # [B, T_local, d_model]
        return self.lm_head(x)  # [B, T_local, vocab_size]


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

    assert (
        args.cp_size <= world_size
    ), f"CP size must be <= World size: cp_size={args.cp_size} > world_size={world_size}"
    assert args.cp_size > 0, f"Context parallelism size must be > 0: cp_size={args.cp_size} <= 0"

    # --- Device setup ---
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    config = GPT_CONFIGS[args.config]
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # --- Context parallelism setup ---
    cp_group = dist.new_group(ranks=list(range(args.cp_size)))
    cp_rank = dist.get_rank(cp_group)

    if rank == 0:
        logger.info(
            "GPT-2 benchmark - config: %s, world_size: %d, cp_size: %d, dtype: %s",
            args.config,
            world_size,
            args.cp_size,
            args.dtype,
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

    model = GPT(config, cp_group).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    mem_model = get_gpu_memory_mb(device)
    n_params = count_parameters(model)

    if rank == 0:
        logger.info("Model params: %s", f"{n_params:,}")
        logger.info("Model size: %.2f MB", mem_model)

    seq_len = min(args.seq_len, config.max_seq_len)
    assert (
        seq_len % args.cp_size == 0
    ), f"Sequence length must be divisible by CP size: seq_len={seq_len} % cp_size={args.cp_size} != 0"
    chunk_len = seq_len // args.cp_size

    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    labels = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)

    # Due to Context Parallelism, each GPU takes its chunk of the sequence.
    # So we need to slice the input_ids, position_ids, and labels accordingly.
    input_ids = input_ids[:, cp_rank * chunk_len : (cp_rank + 1) * chunk_len].contiguous()
    position_ids = position_ids[:, cp_rank * chunk_len : (cp_rank + 1) * chunk_len].contiguous()
    labels = labels[:, cp_rank * chunk_len : (cp_rank + 1) * chunk_len].contiguous()

    for step in range(args.warmup):
        logits = model(input_ids, position_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        # All-reduce a detached copy for logging only - gradients are already
        # correct because ring attention ensures each GPU's backward accounts
        # for all KV blocks via the P2P communication graph.
        loss_avg = loss.detach().clone()
        dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG, group=cp_group)
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
        logits = model(input_ids, position_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        loss_avg = loss.detach().clone()
        dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG, group=cp_group)
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
        mode="cp",
        config=args.config,
        dtype=args.dtype,
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
        out_path = os.path.join(args.output_dir, f"results_train_gpt_cp_{args.config}.json")
        with open(out_path, "w") as fout:
            json.dump(results, fout, indent=2)
        logger.info("=" * 60)
        logger.info(
            "  GPT-2 %s - Context Parallelism (CP=%d) - %d GPU(s) - %s",
            args.config.upper(),
            args.cp_size,
            world_size,
            args.dtype,
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
