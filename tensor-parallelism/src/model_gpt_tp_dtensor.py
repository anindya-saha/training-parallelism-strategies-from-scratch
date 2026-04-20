"""TP GPT-style transformer using PyTorch's DTensor parallel API.

Same model architecture as model_gpt_tp.py but replaces hand-written
ColumnParallelLinear / RowParallelLinear / autograd primitives with
torch.distributed.tensor.parallel (ColwiseParallel, RowwiseParallel,
PrepareModuleInput, SequenceParallel).

The key idea: build a *standard* nn.Module with plain nn.Linear layers,
then call parallelize_module() with a sharding plan. PyTorch handles
weight sharding, communication insertion, and backward gradients.

Run:
    torchrun --nproc_per_node=2 src/model_gpt_tp_dtensor.py
    torchrun --nproc_per_node=4 src/model_gpt_tp_dtensor.py --n-heads 8

Compare output JSON against model_gpt_tp.py for the same config.
"""

import argparse
import json
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    loss_parallel,
    parallelize_module,
)

from utils import count_parameters, get_gpu_memory_mb, get_gpu_peak_memory_mb

logger = logging.getLogger(__name__)

# ================================================================
# Default Model & Benchmark Constants (same as model_gpt_tp.py)
# ================================================================

DEFAULT_D_MODEL = 512
DEFAULT_N_HEADS = 8
DEFAULT_D_FF = 2048
DEFAULT_N_LAYERS = 6
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_MAX_SEQ_LEN = 512

DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 256
DEFAULT_NUM_WARMUP = 3
DEFAULT_NUM_BENCHMARK = 10


# ================================================================
# Standard model components (identical to model_gpt.py)
# ================================================================
# The whole point: we write the model with plain nn.Linear layers.
# No custom autograd functions, no manual sharding, no communication
# code.  parallelize_module() adds all of that after construction.


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        # With use_local_output=False the projections return DTensors
        # sharded on dim -1. Using the *global* n_heads in the view lets
        # DTensor map the shard onto the head dimension automatically.
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # F.scaled_dot_product_attention handles DTensors natively and
        # avoids the reshape-propagation issue that manual Q @ K.T hits.
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=bias)
        self.W2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.gelu(self.W1(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_bias: bool = False,
        ffn_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, bias=attn_bias)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, bias=ffn_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
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
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    attn_bias=attn_bias,
                    ffn_bias=ffn_bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


# ================================================================
# Apply TP via parallelize_module
# ================================================================
# This is the entire "parallelism" code. One function, no custom
# autograd, no manual sharding.
#
# The plan mirrors exactly what model_gpt_tp.py does by hand:
#
#   Attention:
#     W_q, W_k, W_v -> ColwiseParallel  (split output dim across GPUs)
#     W_o           -> RowwiseParallel   (split input dim, all-reduce output)
#
#   FFN:
#     W1 -> ColwiseParallel   (split d_ff across GPUs)
#     W2 -> RowwiseParallel   (split d_ff input, all-reduce output)
#
#   lm_head -> ColwiseParallel (split vocab across GPUs)
#
# Loss Parallel (loss_parallel context manager):
#   lm_head output stays as a DTensor sharded on the vocab dimension.
#   Inside loss_parallel(), F.cross_entropy computes the loss without
#   materializing the full (batch*seq, vocab) logits on any single GPU.
#   Only a small all-reduce for the log-sum-exp denominator is needed,
#   saving both memory and communication vs. a full all-gather.


def apply_tp(model: GPT, mesh) -> GPT:
    for block in model.blocks:
        block_plan = {
            # --- Attention ---
            "attn": PrepareModuleInput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Replicate(),),
            ),
            # use_local_output=False keeps output as a DTensor so that
            # .view(B, T, n_heads, d_head) uses global n_heads and DTensor
            # automatically maps the shard onto the head dimension.
            "attn.W_q": ColwiseParallel(use_local_output=False),
            "attn.W_k": ColwiseParallel(use_local_output=False),
            "attn.W_v": ColwiseParallel(use_local_output=False),
            "attn.W_o": RowwiseParallel(),
            # --- FFN ---
            "ffn": PrepareModuleInput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Replicate(),),
            ),
            "ffn.W1": ColwiseParallel(),
            "ffn.W2": RowwiseParallel(),
        }
        parallelize_module(block, mesh, block_plan)

    # lm_head: ColwiseParallel shards vocab dim across GPUs.
    # use_local_output=False keeps the output as a DTensor so that
    # loss_parallel() can compute cross-entropy on sharded logits.
    parallelize_module(
        model,
        mesh,
        {"lm_head": ColwiseParallel(use_local_output=False)},
    )
    return model


# ================================================================
#  Main
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Tensor parallelism GPT benchmark (DTensor)"
    )
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
        logger.info("TP benchmark (DTensor): %d GPUs", ws)
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

    # 1. Build a plain GPT model (no parallelism baked in)
    model = GPT(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        attn_bias=args.attn_bias,
        ffn_bias=not args.no_ffn_bias,
    ).to(device)

    # 2. Apply TP via DTensor -- this is the only parallelism code
    mesh = init_device_mesh("cuda", (ws,), mesh_dim_names=("tp",))
    apply_tp(model, mesh)

    # foreach=False: Adam's fused _foreach ops require all params to be
    # the same tensor type. With pure TP (no FSDP) the model has a mix of
    # DTensor (parallelized layers) and plain Tensor (LayerNorm, embeddings),
    # so we disable foreach to update each param individually.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
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
        pred = model(input_ids)
        with loss_parallel():
            loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
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
        pred = model(input_ids)
        with loss_parallel():
            loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
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
            mode=f"tp_dtensor_{ws}",
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
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "results_model_gpt_tp_dtensor.json")
        with open(out_path, "w") as fout:
            json.dump(results, fout, indent=2)
        logger.info("=" * 60)
        logger.info(
            "  Model GPT - Tensor Parallelism (DTensor) - %d GPUs", ws
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
