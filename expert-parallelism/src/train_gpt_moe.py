"""GPT-2 with Mixture of Experts -- single-GPU MoE from scratch.

Replaces the dense FFN in each transformer block with a Sparse MoE layer:
  - Router (learned gating) picks top-k experts per token
  - Each expert is an independent FFN with the same architecture
  - Auxiliary loss prevents router collapse (Switch Transformer formulation)
  - Expert capacity limits enforce token overflow when an expert is overloaded

Trains on TinyStories (GPT-2 tokenized memmap files) so we can observe real
learning curves and compare Dense GPT vs MoE GPT convergence.

Three preset configurations:
    --config mini     ~19M params dense / ~57M sparse with 8 experts
    --config small    ~117M params dense / ~352M sparse with 8 experts
    --config medium   ~345M params dense / ~1B sparse with 8 experts

Example:
    torchrun --nproc_per_node=1 src/train_gpt_moe.py --config mini
    torchrun --nproc_per_node=1 src/train_gpt_moe.py --config mini --capacity-factor 0
    torchrun --nproc_per_node=1 src/train_gpt_moe.py --config mini --aux-loss-weight 0
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

from data import create_dataloader
from utils import count_active_parameters, count_parameters, get_gpu_memory_mb, get_gpu_peak_memory_mb

logger = logging.getLogger(__name__)


# ================================================================
# Model Configuration
# ================================================================


@dataclass
class MoEGPTConfig:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 6
    vocab_size: int = 50_257
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = True
    num_experts: int = 8
    top_k: int = 2
    aux_loss_weight: float = 0.01
    capacity_factor: float = 1.25


MOE_CONFIGS = {
    "mini": MoEGPTConfig(
        d_model=512, n_heads=8, d_ff=2048, n_layers=6,
        vocab_size=50_257, max_seq_len=512, dropout=0.1,
        num_experts=8, top_k=2, aux_loss_weight=0.01, capacity_factor=1.25,
    ),
    "small": MoEGPTConfig(
        d_model=768, n_heads=12, d_ff=3072, n_layers=12,
        vocab_size=50_257, max_seq_len=1024, dropout=0.1,
        num_experts=8, top_k=2, aux_loss_weight=0.01, capacity_factor=1.25,
    ),
    "medium": MoEGPTConfig(
        d_model=1024, n_heads=16, d_ff=4096, n_layers=24,
        vocab_size=50_257, max_seq_len=1024, dropout=0.1,
        num_experts=8, top_k=2, aux_loss_weight=0.01, capacity_factor=1.25,
    ),
}


# ================================================================
# Multi-Head Attention
# ================================================================


class Attention(nn.Module):
    def __init__(self, config: MoEGPTConfig):
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
        B, T, _ = x.shape                                          # (B, T, d_model)
        scale = 1.0 / math.sqrt(self.d_head)

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)

        attn = (Q @ K.transpose(-2, -1)) * scale                   # (B, H, T, T)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)                              # (B, H, T, T)
        attn = self.attn_dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.resid_dropout(self.W_o(out))                       # (B, T, d_model)


# ================================================================
# Expert FFN (each expert is one of these)
# ================================================================


class ExpertFFN(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.W2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resid_dropout(self.W2(F.gelu(self.W1(x))))      # (*, d_model)


# ================================================================
# Router (Gating Network)
# ================================================================


class Router(nn.Module):
    """Learned gating network that assigns tokens to top-k experts."""

    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        router_logits = self.gate(x)                                # (B, T, num_experts)
        top_k_logits, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )                                                           # (B, T, top_k) each
        top_k_weights = F.softmax(top_k_logits, dim=-1)            # (B, T, top_k)
        return top_k_weights, top_k_indices, router_logits


# ================================================================
# Sparse MoE Layer
# ================================================================


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts -- replaces the dense FFN in each block."""

    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.experts = nn.ModuleList(
            [ExpertFFN(config) for _ in range(config.num_experts)]
        )
        self.router = Router(config.d_model, config.num_experts, config.top_k)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        B, T, C = x.shape                                          # (B, T, d_model)

        # --- Routing ---
        routing_weights, selected_experts, router_logits = self.router(x)
        # routing_weights:   (B, T, top_k)
        # selected_experts:  (B, T, top_k)
        # router_logits:     (B, T, num_experts)

        x_flat = x.view(-1, C)                                     # (B*T, d_model)
        weights_flat = routing_weights.view(-1, self.top_k)         # (B*T, top_k)
        experts_flat = selected_experts.view(-1, self.top_k)        # (B*T, top_k)

        # --- Expert capacity ---
        use_capacity = self.capacity_factor > 0
        expert_capacity = 0
        if use_capacity:
            total_token_slots = B * T * self.top_k
            expert_capacity = int(
                (total_token_slots / self.num_experts) * self.capacity_factor
            )

        # --- Sparse dispatch ---
        output = torch.zeros_like(x_flat)                           # (B*T, d_model)
        num_dropped = 0

        for i, expert in enumerate(self.experts):
            mask = (experts_flat == i)                              # (B*T, top_k)
            token_mask = mask.any(dim=-1)                           # (B*T,)

            if not token_mask.any():
                continue

            token_indices = token_mask.nonzero(as_tuple=True)[0]    # (num_selected,)

            if use_capacity and len(token_indices) > expert_capacity:
                num_dropped += len(token_indices) - expert_capacity
                token_indices = token_indices[:expert_capacity]
                token_mask = torch.zeros_like(token_mask)
                token_mask[token_indices] = True

            expert_input = x_flat[token_mask]                       # (num_selected, d_model)
            expert_output = expert(expert_input)                    # (num_selected, d_model)

            expert_weights = (weights_flat * mask.float()).sum(dim=-1)  # (B*T,)
            expert_weights = expert_weights[token_mask]                 # (num_selected,)

            output[token_mask] += expert_output * expert_weights.unsqueeze(-1)

        output = output.view(B, T, C)                              # (B, T, d_model)

        # --- Auxiliary loss ---
        aux_loss = self._load_balancing_loss(router_logits, selected_experts)

        return output, aux_loss, num_dropped

    def _load_balancing_loss(
        self, router_logits: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """Switch Transformer auxiliary loss: L_aux = N * sum(f_i * P_i)."""
        B, T, _ = router_logits.shape
        num_tokens = B * T

        flat_experts = selected_experts.view(-1, self.top_k)        # (B*T, top_k)
        expert_mask = torch.zeros(
            num_tokens, self.num_experts, device=router_logits.device
        )                                                           # (B*T, num_experts)
        expert_mask.scatter_(1, flat_experts, 1.0)
        f = expert_mask.mean(dim=0)                                 # (num_experts,)

        P = F.softmax(router_logits, dim=-1).mean(dim=(0, 1))      # (num_experts,)

        aux_loss = self.num_experts * (f * P).sum()
        return aux_loss


# ================================================================
# MoE Transformer Block
# ================================================================


class MoETransformerBlock(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.moe = SparseMoELayer(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        x = x + self.attn(self.ln1(x))                             # (B, T, d_model)
        moe_out, aux_loss, num_dropped = self.moe(self.ln2(x))     # (B, T, d_model)
        x = x + moe_out                                            # (B, T, d_model)
        return x, aux_loss, num_dropped


# ================================================================
# Full MoE GPT Model
# ================================================================


class MoEGPT(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [MoETransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        B, T = input_ids.shape                                      # (B, T)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0) # (1, T)
        x = self.emb_dropout(
            self.tok_emb(input_ids) + self.pos_emb(pos)
        )                                                           # (B, T, d_model)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        total_dropped = 0

        for block in self.blocks:
            x, aux_loss, num_dropped = block(x)
            total_aux_loss = total_aux_loss + aux_loss
            total_dropped += num_dropped

        total_aux_loss = total_aux_loss / self.config.n_layers

        x = self.ln_f(x)                                           # (B, T, d_model)
        logits = self.lm_head(x)                                   # (B, T, vocab_size)
        return logits, total_aux_loss, total_dropped


# ================================================================
# Dense GPT Model (for comparison)
# ================================================================


class FFN(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.W2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resid_dropout(self.W2(F.gelu(self.W1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DenseGPT(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
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
# Training Loop
# ================================================================


@torch.no_grad()
def evaluate(model, val_loader, config, device, is_moe: bool):
    model.eval()
    total_loss = 0.0
    total_aux = 0.0
    count = 0
    max_eval_batches = 20

    for i, (input_ids, labels) in enumerate(val_loader):
        if i >= max_eval_batches:
            break
        input_ids = input_ids.to(device)                            # (B, T)
        labels = labels.to(device)                                  # (B, T)
        if is_moe:
            logits, aux_loss, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size), labels.view(-1)
            )
            total_aux += aux_loss.item()
        else:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size), labels.view(-1)
            )
        total_loss += loss.item()
        count += 1

    model.train()
    avg_loss = total_loss / max(count, 1)
    avg_aux = total_aux / max(count, 1)
    return avg_loss, avg_aux


def parse_args():
    p = argparse.ArgumentParser(description="MoE GPT-2 training on TinyStories")
    p.add_argument(
        "--config", type=str, default="mini",
        choices=list(MOE_CONFIGS.keys()),
    )
    p.add_argument("--mode", type=str, default="moe", choices=["moe", "dense"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--data-dir", type=str, default="../")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--num-experts", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--aux-loss-weight", type=float, default=None)
    p.add_argument("--capacity-factor", type=float, default=None)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    config = MOE_CONFIGS[args.config]
    if args.num_experts is not None:
        config.num_experts = args.num_experts
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.aux_loss_weight is not None:
        config.aux_loss_weight = args.aux_loss_weight
    if args.capacity_factor is not None:
        config.capacity_factor = args.capacity_factor

    seq_len = min(args.seq_len, config.max_seq_len)
    is_moe = args.mode == "moe"

    if rank == 0:
        logger.info("=" * 60)
        logger.info("  %s GPT-2 %s -- TinyStories", "MoE" if is_moe else "Dense", args.config)
        logger.info("=" * 60)
        logger.info(
            "d_model=%d  n_heads=%d  d_ff=%d  n_layers=%d  vocab=%d",
            config.d_model, config.n_heads, config.d_ff,
            config.n_layers, config.vocab_size,
        )
        if is_moe:
            logger.info(
                "num_experts=%d  top_k=%d  aux_loss_weight=%.4f  capacity_factor=%.2f",
                config.num_experts, config.top_k,
                config.aux_loss_weight, config.capacity_factor,
            )

    # --- Model ---
    if is_moe:
        model = MoEGPT(config).to(device)
    else:
        model = DenseGPT(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    sparse_params = count_parameters(model)
    active_params = (
        count_active_parameters(model, config.num_experts, config.top_k)
        if is_moe else sparse_params
    )
    mem_model = get_gpu_memory_mb(device)

    if rank == 0:
        logger.info("Sparse params: %s", f"{sparse_params:,}")
        logger.info("Active params: %s", f"{active_params:,}")
        logger.info("Model memory:  %.1f MB", mem_model)

    # --- Data ---
    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "validation.bin")
    train_loader, train_sampler = create_dataloader(
        train_path, seq_len, args.batch_size,
    )
    val_loader, _ = create_dataloader(
        val_path, seq_len, args.batch_size, shuffle=False,
    )
    train_iter = iter(train_loader)

    if rank == 0:
        logger.info("Training for %d steps, eval every %d steps", args.num_steps, args.eval_interval)
        logger.info("-" * 60)

    # --- Training ---
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    total_tokens = 0
    t_start = time.perf_counter()
    history = []

    for step in range(1, args.num_steps + 1):
        try:
            input_ids, labels = next(train_iter)
        except StopIteration:
            if train_sampler is not None:
                train_sampler.set_epoch(step)
            train_iter = iter(train_loader)
            input_ids, labels = next(train_iter)

        input_ids = input_ids.to(device)                            # (B, T)
        labels = labels.to(device)                                  # (B, T)

        optimizer.zero_grad()

        if is_moe:
            logits, aux_loss, num_dropped = model(input_ids)        # (B, T, V)
            lm_loss = F.cross_entropy(
                logits.view(-1, config.vocab_size), labels.view(-1)
            )
            total_loss = lm_loss + config.aux_loss_weight * aux_loss
        else:
            logits = model(input_ids)                               # (B, T, V)
            lm_loss = F.cross_entropy(
                logits.view(-1, config.vocab_size), labels.view(-1)
            )
            total_loss = lm_loss
            aux_loss = torch.tensor(0.0)
            num_dropped = 0

        total_loss.backward()
        optimizer.step()

        total_tokens += input_ids.numel()
        tokens_per_sec = total_tokens / (time.perf_counter() - t_start)

        total_token_slots = (
            input_ids.shape[0] * input_ids.shape[1] * config.top_k * config.n_layers
            if is_moe else 1
        )
        drop_rate = num_dropped / max(total_token_slots, 1) * 100

        step_record = {
            "step": step,
            "train_loss": round(lm_loss.item(), 4),
            "aux_loss": round(aux_loss.item(), 4) if torch.is_tensor(aux_loss) else 0.0,
            "total_loss": round(total_loss.item(), 4),
            "tokens_dropped": num_dropped,
            "drop_rate_pct": round(drop_rate, 2),
        }
        history.append(step_record)

        if rank == 0 and step % 10 == 0:
            logger.info(
                "step %4d/%d  lm=%.4f  aux=%.4f  total=%.4f  dropped=%d (%.1f%%)  tok/s=%.0f",
                step, args.num_steps, lm_loss.item(),
                aux_loss.item() if torch.is_tensor(aux_loss) else 0.0,
                total_loss.item(), num_dropped, drop_rate, tokens_per_sec,
            )

        if step % args.eval_interval == 0 or step == args.num_steps:
            val_loss, val_aux = evaluate(model, val_loader, config, device, is_moe)
            if rank == 0:
                logger.info(
                    "  [eval] step %d  val_loss=%.4f  val_aux=%.4f",
                    step, val_loss, val_aux,
                )
            step_record["val_loss"] = round(val_loss, 4)
            step_record["val_aux_loss"] = round(val_aux, 4)

    # --- Results ---
    peak = get_gpu_peak_memory_mb(device)
    elapsed = time.perf_counter() - t_start

    if rank == 0:
        results = {
            "mode": args.mode,
            "config": args.config,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "d_ff": config.d_ff,
            "n_layers": config.n_layers,
            "vocab_size": config.vocab_size,
            "num_experts": config.num_experts if is_moe else 0,
            "top_k": config.top_k if is_moe else 0,
            "aux_loss_weight": config.aux_loss_weight if is_moe else 0,
            "capacity_factor": config.capacity_factor if is_moe else 0,
            "batch_size": args.batch_size,
            "seq_len": seq_len,
            "num_steps": args.num_steps,
            "sparse_params": sparse_params,
            "active_params": active_params,
            "mem_model_mb": round(mem_model, 2),
            "mem_peak_mb": round(peak, 2),
            "elapsed_sec": round(elapsed, 2),
            "tokens_per_sec": round(total_tokens / elapsed, 1),
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1].get("val_loss", None),
            "history": history,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        suffix = "moe" if is_moe else "dense"
        out_path = os.path.join(
            args.output_dir, f"results_train_gpt_{suffix}_{args.config}.json"
        )
        with open(out_path, "w") as fout:
            json.dump(results, fout, indent=2)

        logger.info("=" * 60)
        logger.info("  %s GPT-2 %s -- Summary", "MoE" if is_moe else "Dense", args.config.upper())
        logger.info("=" * 60)
        logger.info("  %-25s %12s", "Sparse Params", f"{sparse_params:,}")
        logger.info("  %-25s %12s", "Active Params", f"{active_params:,}")
        logger.info("  %-25s %12.1f", "Model MB", mem_model)
        logger.info("  %-25s %12.1f", "Peak MB", peak)
        logger.info("  %-25s %12.1f", "Tokens/sec", total_tokens / elapsed)
        logger.info("  %-25s %12.4f", "Final Train Loss", history[-1]["train_loss"])
        if history[-1].get("val_loss") is not None:
            logger.info("  %-25s %12.4f", "Final Val Loss", history[-1]["val_loss"])
        logger.info("  %-25s %s", "Output", out_path)
        logger.info("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
