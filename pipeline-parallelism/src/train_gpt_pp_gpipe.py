"""GPT-2 with hand-written Pipeline Parallelism -- GPipe schedule.

GPipe schedule: split the global batch into M microbatches. All microbatch
forwards flow through the entire pipeline first, then all backwards run in
reverse microbatch order. Gradients accumulate over microbatches so the
effective batch size matches the global batch.

Generalized to N stages (--pp-size). Each rank owns a contiguous slice of the
model's flat layer list. During the forward phase, each rank stores all M
activation tensors until the backward phase drains them.

Peak stored activations on any rank = M (the number of microbatches).

Example:
    torchrun --nproc_per_node=2 src/train_gpt_pp_gpipe.py --config mini --pp-size 2
    torchrun --nproc_per_node=4 src/train_gpt_pp_gpipe.py --config mini --pp-size 4 --num-microbatches 8
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
# Model Components
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
    """GPT with flat nn.ModuleList for easy pipeline slicing.

    layers = [TokPosEmbedding, Block_0, ..., Block_{n-1}, LMHead]
    Total modules = n_layers + 2
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
# Pipeline Partitioning
# ================================================================


def get_stage(
    config: GPTConfig, rank: int, pp_size: int, device: torch.device
) -> tuple[nn.Sequential, int, int]:
    """Partition model into pipeline stages."""
    with torch.device("meta"):
        full_model = GPT(config)

    all_layers = list(full_model.layers)
    n_mod = len(all_layers)
    chunk = (n_mod + pp_size - 1) // pp_size
    start = rank * chunk
    end = min(start + chunk, n_mod)
    local_layers = all_layers[start:end]

    stage = nn.Sequential(*local_layers)
    stage = stage.to_empty(device=device)
    stage.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    return stage, start, end


def lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Causal LM loss: predict next token."""
    vocab = logits.size(-1)
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, vocab),
        input_ids[:, 1:].contiguous().view(-1),
    )


# ================================================================
# GPipe Pipeline Schedule
# ================================================================


def pipeline_forward_backward_gpipe(
    stage: nn.Sequential,
    rank: int,
    pp_size: int,
    micro_inputs: list[torch.Tensor],
    d_model: int,
    device: torch.device,
) -> torch.Tensor | None:
    """GPipe schedule: all microbatch forwards, then all microbatch backwards.

    Args:
        micro_inputs: list of M microbatch input_ids tensors, each (mbs, T).

    Returns loss tensor (sum over microbatches) on last rank, None otherwise.
    """
    n_micro = len(micro_inputs)
    mbs, T = micro_inputs[0].shape
    is_first = rank == 0
    is_last = rank == pp_size - 1

    # Storage for activations needed during backward
    saved_outputs: list[torch.Tensor] = []  # outputs of this stage's forward (need grad)
    saved_inputs: list[torch.Tensor] = []  # inputs to this stage that require grad (not first)
    last_stage_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    # ========== Forward Phase ==========
    for b in range(n_micro):
        if is_first:
            h = stage(micro_inputs[b])  # (mbs, T, d_model)
            saved_outputs.append(h)
            dist.send(h.detach().contiguous(), dst=rank + 1)
        elif is_last:
            buf = torch.empty(mbs, T, d_model, device=device)
            dist.recv(buf, src=rank - 1)
            h_req = buf.clone().requires_grad_(True)
            logits = stage(h_req)  # (mbs, T, vocab)
            last_stage_data.append((h_req, logits, micro_inputs[b]))
        else:
            buf = torch.empty(mbs, T, d_model, device=device)
            dist.recv(buf, src=rank - 1)
            h_req = buf.clone().requires_grad_(True)
            h = stage(h_req)  # (mbs, T, d_model)
            saved_outputs.append(h)
            saved_inputs.append(h_req)
            dist.send(h.detach().contiguous(), dst=rank + 1)

    # ========== Backward Phase (reverse microbatch order) ==========
    total_loss = torch.tensor(0.0, device=device)

    for b in range(n_micro - 1, -1, -1):
        if is_last:
            h_req, logits, ids_b = last_stage_data[b]
            loss_b = lm_loss(logits, ids_b)
            loss_b.backward()
            total_loss += loss_b.detach()
            assert h_req.grad is not None
            dist.send(h_req.grad.detach().contiguous(), dst=rank - 1)
        elif is_first:
            grad_h = torch.empty(mbs, T, d_model, device=device)
            dist.recv(grad_h, src=rank + 1)
            saved_outputs[b].backward(grad_h)
        else:
            grad_h = torch.empty(mbs, T, d_model, device=device)
            dist.recv(grad_h, src=rank + 1)
            saved_outputs[b].backward(grad_h)
            assert saved_inputs[b].grad is not None
            dist.send(saved_inputs[b].grad.detach().contiguous(), dst=rank - 1)

    if is_last:
        return total_loss / n_micro
    return None


# ================================================================
# Main
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(description="GPT-2 Pipeline Parallelism - GPipe Schedule")
    p.add_argument(
        "--config",
        type=str,
        default="mini",
        choices=list(GPT_CONFIGS.keys()),
        help="Model configuration: mini, small, medium",
    )
    p.add_argument("--pp-size", type=int, default=2, help="Pipeline parallelism size (num stages)")
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

    if rank == 0:
        logger.info(
            "GPT-2 Pipeline Parallelism (GPipe) -- config: %s, pp_size: %d, n_micro: %d, mbs: %d",
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

    stage, start_idx, end_idx = get_stage(config, rank, pp_size, device)
    optimizer = torch.optim.Adam(stage.parameters(), lr=1e-4)

    mem_model = get_gpu_memory_mb(device)
    n_params = count_parameters(stage)

    if rank == 0:
        logger.info("Rank 0 stage params: %s (layers [%d:%d])", f"{n_params:,}", start_idx, end_idx)

    seq_len = min(args.seq_len, config.max_seq_len)

    # Synthetic data -- broadcast from rank 0
    if rank == 0:
        input_ids = torch.randint(
            0, config.vocab_size, (args.batch_size, seq_len), device=device
        )
    else:
        input_ids = torch.empty(args.batch_size, seq_len, device=device, dtype=torch.long)
    dist.broadcast(input_ids, src=0)

    def split_microbatches(ids: torch.Tensor) -> list[torch.Tensor]:
        return [ids[b * mbs : (b + 1) * mbs].contiguous() for b in range(n_micro)]

    # --- Warmup ---
    for _ in range(args.warmup):
        optimizer.zero_grad()
        micro_inputs = split_microbatches(input_ids)
        pipeline_forward_backward_gpipe(
            stage, rank, pp_size, micro_inputs, config.d_model, device
        )
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
        micro_inputs = split_microbatches(input_ids)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss = pipeline_forward_backward_gpipe(
            stage, rank, pp_size, micro_inputs, config.d_model, device
        )

        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if loss is not None:
            last_loss = loss.detach()
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
        mode="pp_gpipe",
        config=args.config,
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
            f"results_train_gpt_pp_gpipe_{args.config}_pp{pp_size}_m{n_micro}.json",
        )
        with open(out_path, "w") as fout:
            json.dump(results, fout, indent=2)
        logger.info("=" * 60)
        logger.info(
            "  GPT-2 %s - Pipeline Parallelism (PP=%d, GPipe, M=%d) - %d GPU(s)",
            args.config.upper(),
            pp_size,
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
