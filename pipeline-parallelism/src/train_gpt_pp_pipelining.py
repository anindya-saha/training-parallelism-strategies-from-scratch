"""GPT-2 with PyTorch torch.distributed.pipelining -- framework-managed PP.

Uses torch.distributed.pipelining (PyTorch 2.7+) to handle all pipeline
communication automatically. The model is the same GPT with a flat layer list,
but instead of manual send/recv and microbatch loops, we wrap each stage in a
PipelineStage and pick a schedule (ScheduleGPipe or Schedule1F1B).

Key contrast with the hand-written scripts:
    - No manual dist.send/recv
    - No microbatch splitting logic
    - No activation storage management
    - Schedule selection is one line

The model code is identical to the single-GPU version. Parallelism is applied
externally via PipelineStage + schedule.

Example:
    torchrun --nproc_per_node=2 src/train_gpt_pp_pipelining.py --config mini --pp-size 2 --schedule gpipe
    torchrun --nproc_per_node=4 src/train_gpt_pp_pipelining.py --config mini --pp-size 4 --schedule 1f1b
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
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

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
    n_layers: int = 8
    vocab_size: int = 10_000
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = True


GPT_CONFIGS = {
    "mini": GPTConfig(
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=8,
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
# Model Components (identical to hand-written scripts)
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

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape  # (B, T, d_model)
        scale = 1.0 / math.sqrt(self.d_head)

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)

        attn = (Q @ K.transpose(-2, -1)) * scale  # (B, H, T, T)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.resid_dropout(self.W_o(out))


class FFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.W2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resid_dropout(self.W2(F.gelu(self.W1(x))))  # (B, T, d_model)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # (B, T, d_model)
        x = x + self.ffn(self.ln2(x))  # (B, T, d_model)
        return x


class TokPosEmbedding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape  # (B, T)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        return self.emb_dropout(self.tok_emb(input_ids) + self.pos_emb(pos))  # (B, T, d_model)


class LMHead(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.d_model)
        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.ln_f(x))  # (B, T, vocab_size)


# ================================================================
# Full GPT Model (flat layer list for pipeline partitioning)
# ================================================================


class GPT(nn.Module):
    """GPT with flat nn.ModuleList for pipeline slicing.

    The model is parallelism-agnostic. PipelineStage wraps it externally.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        parts: list[nn.Module] = [TokPosEmbedding(config)]
        for _ in range(config.n_layers):
            parts.append(TransformerBlock(config))
        parts.append(LMHead(config))
        self.layers = nn.ModuleList(parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ================================================================
# Pipeline Stage Construction (using torch.distributed.pipelining)
# ================================================================


def build_pipeline_stage(
    config: GPTConfig,
    rank: int,
    pp_size: int,
    device: torch.device,
    batch_size: int,
    seq_len: int,
) -> tuple[PipelineStage, nn.Sequential, int, int]:
    """Build a PipelineStage for this rank using manual model partitioning.

    Same partitioning logic as the hand-written scripts: build on meta,
    slice layers, materialize on device. Then wrap in PipelineStage which
    handles all communication buffer allocation and send/recv ops.

    Returns (pipeline_stage, stage_module, start_idx, end_idx).
    """
    with torch.device("meta"):
        full_model = GPT(config)

    all_layers = list(full_model.layers)
    n_mod = len(all_layers)
    chunk = (n_mod + pp_size - 1) // pp_size
    start = rank * chunk
    end = min(start + chunk, n_mod)
    local_layers = all_layers[start:end]

    stage_module = nn.Sequential(*local_layers)
    stage_module = stage_module.to_empty(device=device)
    stage_module.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )

    is_first = rank == 0
    is_last = rank == pp_size - 1

    # PipelineStage needs example input shapes for buffer allocation.
    # First stage receives input_ids (int64); other stages receive hidden states (float).
    if is_first:
        example_input = (torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device),)
    else:
        example_input = (torch.randn(batch_size, seq_len, config.d_model, device=device),)

    # output_args needed if output shape/dtype differs from default inference
    if is_last:
        output_args = (torch.randn(batch_size, seq_len, config.vocab_size, device=device),)
    else:
        output_args = (torch.randn(batch_size, seq_len, config.d_model, device=device),)

    pipeline_stage = PipelineStage(
        stage_module,
        stage_index=rank,
        num_stages=pp_size,
        device=device,
        input_args=example_input,
        output_args=output_args,
    )

    return pipeline_stage, stage_module, start, end


def lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Causal LM loss: predict next token."""
    vocab = logits.size(-1)
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, vocab),
        input_ids[:, 1:].contiguous().view(-1),
    )


# ================================================================
# Main
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="GPT-2 Pipeline Parallelism - torch.distributed.pipelining"
    )
    p.add_argument(
        "--config",
        type=str,
        default="mini",
        choices=list(GPT_CONFIGS.keys()),
        help="Model configuration: mini, small, medium",
    )
    p.add_argument("--pp-size", type=int, default=2, help="Pipeline parallelism size (num stages)")
    p.add_argument(
        "--schedule",
        type=str,
        default="gpipe",
        choices=["gpipe", "1f1b"],
        help="Pipeline schedule: gpipe or 1f1b",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--num-microbatches", type=int, default=4, help="Number of microbatches (M)")
    p.add_argument("--warmup", type=int, default=DEFAULT_NUM_WARMUP)
    p.add_argument("--benchmark", type=int, default=DEFAULT_NUM_BENCHMARK)
    p.add_argument("--output-dir", type=str, default="outputs")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%H:%M:%S]")
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

    pp_size = args.pp_size
    assert pp_size == world_size, (
        f"For pure PP, --pp-size must equal world_size: pp_size={pp_size}, world_size={world_size}"
    )
    assert args.batch_size % args.num_microbatches == 0, (
        f"batch_size must be divisible by num_microbatches: "
        f"{args.batch_size} % {args.num_microbatches} != 0"
    )

    config = GPT_CONFIGS[args.config]
    n_micro = args.num_microbatches
    mbs = args.batch_size // n_micro
    seq_len = min(args.seq_len, config.max_seq_len)
    is_first = rank == 0
    is_last = rank == pp_size - 1

    if rank == 0:
        logger.info(
            "GPT-2 Pipeline Parallelism (pipelining, %s) -- config: %s, pp_size: %d, "
            "n_micro: %d, mbs: %d",
            args.schedule,
            args.config,
            pp_size,
            n_micro,
            mbs,
        )
        logger.info(
            "d_model=%d, n_heads=%d, d_ff=%d, n_layers=%d, vocab=%d",
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.n_layers,
            config.vocab_size,
        )

    # Build pipeline stage using torch.distributed.pipelining
    pipeline_stage, stage_module, start_idx, end_idx = build_pipeline_stage(
        config, rank, pp_size, device, mbs, seq_len
    )

    optimizer = torch.optim.Adam(stage_module.parameters(), lr=1e-4)

    mem_model = get_gpu_memory_mb(device)
    n_params = count_parameters(stage_module)

    if rank == 0:
        logger.info("Rank 0 stage params: %s (layers [%d:%d])", f"{n_params:,}", start_idx, end_idx)

    # Create schedule
    if args.schedule == "gpipe":
        schedule = ScheduleGPipe(pipeline_stage, n_microbatches=n_micro, loss_fn=lm_loss)
    else:
        schedule = Schedule1F1B(pipeline_stage, n_microbatches=n_micro, loss_fn=lm_loss)

    # Synthetic data -- broadcast from rank 0
    if rank == 0:
        input_ids = torch.randint(
            0, config.vocab_size, (args.batch_size, seq_len), device=device
        )
    else:
        input_ids = torch.empty(args.batch_size, seq_len, device=device, dtype=torch.long)
    dist.broadcast(input_ids, src=0)

    # --- Warmup ---
    for _ in range(args.warmup):
        optimizer.zero_grad()
        if is_first:
            schedule.step(input_ids)
        elif is_last:
            losses: list[torch.Tensor] = []
            schedule.step(target=input_ids, losses=losses)
        else:
            schedule.step()
        optimizer.step()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # --- Benchmark ---
    if rank == 0:
        logger.info("--- Benchmark (%d steps) ---", args.benchmark)

    step_t_list = []
    last_loss = torch.tensor(0.0, device=device)

    for step in range(args.benchmark):
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if is_first:
            schedule.step(input_ids)
        elif is_last:
            losses = []
            schedule.step(target=input_ids, losses=losses)
        else:
            schedule.step()

        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Gather loss from last rank
        if is_last and losses:
            last_loss = sum(l.detach() for l in losses) / len(losses)
        dist.broadcast(last_loss, src=pp_size - 1)

        step_t_list.append(t1 - t0)

        if rank == 0:
            logger.info(
                "  step %d/%d  loss=%.4f  step=%.1fms",
                step + 1,
                args.benchmark,
                last_loss.item(),
                (t1 - t0) * 1000,
            )

    peak = get_gpu_peak_memory_mb(device)

    average = lambda values: sum(values) / len(values)
    results = dict(
        mode=f"pp_pipelining_{args.schedule}",
        config=args.config,
        schedule=args.schedule,
        rank=rank,
        num_gpus=world_size,
        pp_size=pp_size,
        num_microbatches=n_micro,
        microbatch_size=mbs,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        vocab_size=config.vocab_size,
        batch_size=args.batch_size,
        seq_len=seq_len,
        params_per_gpu=n_params,
        mem_model_mb=round(mem_model, 2),
        mem_peak_mb=round(peak, 2),
        step_ms=round(average(step_t_list) * 1000, 3),
        tokens_per_sec=round(args.batch_size * seq_len / average(step_t_list), 1),
        loss=round(last_loss.item(), 4),
    )

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(
            args.output_dir,
            f"results_train_gpt_pp_pipelining_{args.schedule}_{args.config}_pp{pp_size}_m{n_micro}.json",
        )
        with open(out_path, "w") as fout:
            json.dump(results, fout, indent=2)
        logger.info("=" * 60)
        logger.info(
            "  GPT-2 %s - torch.distributed.pipelining (PP=%d, %s, M=%d) - %d GPU(s)",
            args.config.upper(),
            pp_size,
            args.schedule.upper(),
            n_micro,
            world_size,
        )
        logger.info("=" * 60)
        for k in [
            "params_per_gpu",
            "mem_model_mb",
            "mem_peak_mb",
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
        logger.info("  %-25s %s", "Output", out_path)
        logger.info("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
