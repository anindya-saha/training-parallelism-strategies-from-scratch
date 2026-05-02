"""GPT-2 with Expert Parallelism -- hand-written all-to-all dispatch.

Distributes MoE experts across GPUs using all-to-all communication:
  1. Router runs on ALL tokens on ALL GPUs (replicated)
  2. Tokens are permuted by target expert and dispatched via all-to-all
  3. Each GPU runs its local experts on received tokens
  4. Results are returned via all-to-all and unpermuted to original order

Supports EP+DP composition on a 2D process group mesh:
  - EP dimension: experts sharded, tokens dispatched via all-to-all
  - DP dimension: model replicated, gradients synced via all-reduce

Example:
    # Pure EP: 4 GPUs, all in one EP group
    torchrun --nproc_per_node=4 src/train_gpt_ep.py --config mini --ep-size 4

    # EP+DP: 4 GPUs, ep_size=2 (2 EP groups), dp_size=2 (2 DP replicas)
    torchrun --nproc_per_node=4 src/train_gpt_ep.py --config mini --ep-size 2
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
# Model Configuration (same as Part 7)
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
    capacity_factor: float = 0.0


MOE_CONFIGS = {
    "mini": MoEGPTConfig(
        d_model=512, n_heads=8, d_ff=2048, n_layers=6,
        vocab_size=50_257, max_seq_len=512, dropout=0.1,
        num_experts=8, top_k=2, aux_loss_weight=0.01, capacity_factor=0.0,
    ),
    "small": MoEGPTConfig(
        d_model=768, n_heads=12, d_ff=3072, n_layers=12,
        vocab_size=50_257, max_seq_len=1024, dropout=0.1,
        num_experts=8, top_k=2, aux_loss_weight=0.01, capacity_factor=0.0,
    ),
    "medium": MoEGPTConfig(
        d_model=1024, n_heads=16, d_ff=4096, n_layers=24,
        vocab_size=50_257, max_seq_len=1024, dropout=0.1,
        num_experts=8, top_k=2, aux_loss_weight=0.01, capacity_factor=0.0,
    ),
}


# ================================================================
# Multi-Head Attention (same as Part 7)
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
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.resid_dropout(self.W_o(out))


# ================================================================
# Expert FFN
# ================================================================


class ExpertFFN(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.W2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resid_dropout(self.W2(F.gelu(self.W1(x))))


# ================================================================
# Router
# ================================================================


class Router(nn.Module):
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
# All-to-All Token Dispatch / Combine
# ================================================================


def all_to_all_dispatch(
    x_flat: torch.Tensor,
    expert_indices: torch.Tensor,
    ep_size: int,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], torch.Tensor]:
    """Permute tokens by expert, then all-to-all dispatch to expert-owning GPUs.

    Args:
        x_flat: (N, d_model) flattened token embeddings (N = B*T*top_k)
        expert_indices: (N,) global expert index for each token-slot
        ep_size: number of GPUs in EP group
        num_experts: total number of experts across all GPUs
        ep_group: process group for EP communication

    Returns:
        recv_tokens: tokens received for this GPU's local experts
        sort_indices: for later unpermute
        input_splits: how many tokens sent to each rank
        output_splits: how many tokens received from each rank
        recv_expert_ids: local expert index for each received token
    """
    experts_per_rank = num_experts // ep_size
    target_rank = expert_indices // experts_per_rank                 # (N,)

    sort_indices = torch.argsort(target_rank, stable=True)          # (N,)
    x_sorted = x_flat[sort_indices]                                 # (N, d_model)
    expert_ids_sorted = expert_indices[sort_indices]                 # (N,)

    # Compute how many tokens go to each rank
    input_splits = [
        (target_rank[sort_indices] == r).sum().item() for r in range(ep_size)
    ]

    # Exchange split counts so each rank knows how many to receive
    input_splits_tensor = torch.tensor(input_splits, dtype=torch.long, device=x_flat.device)
    output_splits_tensor = torch.empty_like(input_splits_tensor)
    dist.all_to_all_single(
        output_splits_tensor, input_splits_tensor, group=ep_group
    )
    output_splits = output_splits_tensor.tolist()

    # All-to-all: send tokens to the GPU that owns their target expert
    total_recv = sum(output_splits)
    recv_tokens = torch.empty(
        total_recv, x_flat.shape[1], dtype=x_flat.dtype, device=x_flat.device
    )                                                               # (total_recv, d_model)
    dist.all_to_all_single(
        recv_tokens, x_sorted, output_splits, input_splits, group=ep_group
    )

    # Also exchange expert IDs so we know which local expert to run
    recv_expert_ids = torch.empty(
        total_recv, dtype=expert_ids_sorted.dtype, device=x_flat.device
    )
    dist.all_to_all_single(
        recv_expert_ids, expert_ids_sorted, output_splits, input_splits, group=ep_group
    )
    # Convert global expert ID to local expert ID
    ep_rank = dist.get_rank(ep_group)
    recv_expert_ids = recv_expert_ids - ep_rank * experts_per_rank  # (total_recv,)

    return recv_tokens, sort_indices, input_splits, output_splits, recv_expert_ids


def all_to_all_combine(
    expert_output: torch.Tensor,
    sort_indices: torch.Tensor,
    input_splits: list[int],
    output_splits: list[int],
    ep_group: dist.ProcessGroup,
    original_size: int,
) -> torch.Tensor:
    """Reverse the all-to-all dispatch: return expert outputs to originating GPUs.

    Args:
        expert_output: (total_recv, d_model) outputs from local experts
        sort_indices: from dispatch, used to unpermute
        input_splits: from dispatch (now becomes output_splits for reverse)
        output_splits: from dispatch (now becomes input_splits for reverse)
        ep_group: process group for EP communication
        original_size: N = B*T*top_k, the original number of token-slots

    Returns:
        combined: (N, d_model) expert outputs in original token order
    """
    recv_back = torch.empty(
        original_size, expert_output.shape[1],
        dtype=expert_output.dtype, device=expert_output.device,
    )                                                               # (N, d_model)
    dist.all_to_all_single(
        recv_back, expert_output, input_splits, output_splits, group=ep_group
    )

    # Unpermute to restore original token order
    combined = torch.empty_like(recv_back)                          # (N, d_model)
    combined[sort_indices] = recv_back
    return combined


# ================================================================
# Distributed Sparse MoE Layer
# ================================================================


class DistributedSparseMoELayer(nn.Module):
    """MoE layer with all-to-all expert parallelism.

    Each GPU holds num_experts // ep_size local experts. Tokens are
    dispatched to the GPU owning their target expert via all-to-all.
    """

    def __init__(self, config: MoEGPTConfig, ep_size: int, ep_group: dist.ProcessGroup):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.ep_size = ep_size
        self.ep_group = ep_group

        assert config.num_experts % ep_size == 0
        self.experts_per_rank = config.num_experts // ep_size

        self.experts = nn.ModuleList(
            [ExpertFFN(config) for _ in range(self.experts_per_rank)]
        )
        self.router = Router(config.d_model, config.num_experts, config.top_k)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape                                          # (B, T, d_model)

        # --- Routing (replicated on all GPUs) ---
        routing_weights, selected_experts, router_logits = self.router(x)
        # routing_weights:   (B, T, top_k)
        # selected_experts:  (B, T, top_k) -- global expert IDs

        weights_flat = routing_weights.view(-1, self.top_k)         # (B*T, top_k)
        experts_flat = selected_experts.view(-1, self.top_k)        # (B*T, top_k)

        # Expand tokens for each top-k slot
        x_flat = x.view(-1, C)                                     # (B*T, d_model)
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.top_k, -1)  # (B*T, top_k, d_model)
        x_expanded = x_expanded.reshape(-1, C)                     # (B*T*top_k, d_model)
        expert_indices = experts_flat.reshape(-1)                   # (B*T*top_k,)
        weights_expanded = weights_flat.reshape(-1)                 # (B*T*top_k,)

        N = x_expanded.shape[0]                                     # B*T*top_k

        # --- Dispatch: all-to-all send tokens to expert-owning GPU ---
        recv_tokens, sort_indices, in_splits, out_splits, local_expert_ids = \
            all_to_all_dispatch(
                x_expanded, expert_indices, self.ep_size,
                self.num_experts, self.ep_group,
            )

        # --- Local expert compute ---
        total_recv = recv_tokens.shape[0]
        expert_output = torch.zeros_like(recv_tokens)               # (total_recv, d_model)

        for i, expert in enumerate(self.experts):
            mask = (local_expert_ids == i)                          # (total_recv,)
            if not mask.any():
                continue
            expert_output[mask] = expert(recv_tokens[mask])         # (n_i, d_model)

        # --- Combine: all-to-all return results to originating GPU ---
        combined = all_to_all_combine(
            expert_output, sort_indices, in_splits, out_splits,
            self.ep_group, N,
        )                                                           # (B*T*top_k, d_model)

        # --- Weighted sum across top-k slots ---
        combined = combined * weights_expanded.unsqueeze(-1)        # (B*T*top_k, d_model)
        combined = combined.view(B * T, self.top_k, C)              # (B*T, top_k, d_model)
        output = combined.sum(dim=1)                                # (B*T, d_model)
        output = output.view(B, T, C)                               # (B, T, d_model)

        # --- Auxiliary loss (all-reduce across EP group for consistency) ---
        aux_loss = self._load_balancing_loss(router_logits, selected_experts)
        dist.all_reduce(aux_loss, op=dist.ReduceOp.AVG, group=self.ep_group)

        return output, aux_loss

    def _load_balancing_loss(
        self, router_logits: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = router_logits.shape
        num_tokens = B * T

        flat_experts = selected_experts.view(-1, self.top_k)
        expert_mask = torch.zeros(
            num_tokens, self.num_experts, device=router_logits.device
        )
        expert_mask.scatter_(1, flat_experts, 1.0)
        f = expert_mask.mean(dim=0)                                 # (num_experts,)
        P = F.softmax(router_logits, dim=-1).mean(dim=(0, 1))      # (num_experts,)
        aux_loss = self.num_experts * (f * P).sum()
        return aux_loss


# ================================================================
# MoE Transformer Block (EP version)
# ================================================================


class MoETransformerBlock(nn.Module):
    def __init__(self, config: MoEGPTConfig, ep_size: int, ep_group: dist.ProcessGroup):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.moe = DistributedSparseMoELayer(config, ep_size, ep_group)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))                             # (B, T, d_model)
        moe_out, aux_loss = self.moe(self.ln2(x))                  # (B, T, d_model)
        x = x + moe_out
        return x, aux_loss


# ================================================================
# Full MoE GPT Model (EP version)
# ================================================================


class MoEGPT(nn.Module):
    def __init__(self, config: MoEGPTConfig, ep_size: int, ep_group: dist.ProcessGroup):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [MoETransformerBlock(config, ep_size, ep_group) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape                                      # (B, T)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0) # (1, T)
        x = self.emb_dropout(
            self.tok_emb(input_ids) + self.pos_emb(pos)
        )                                                           # (B, T, d_model)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss
        total_aux_loss = total_aux_loss / self.config.n_layers

        x = self.ln_f(x)                                           # (B, T, d_model)
        logits = self.lm_head(x)                                   # (B, T, vocab_size)
        return logits, total_aux_loss


# ================================================================
# DP Gradient Sync
# ================================================================


def sync_dp_gradients(model: nn.Module, dp_group: dist.ProcessGroup, ep_size: int):
    """All-reduce gradients for non-expert parameters across DP group.

    Expert weights are unique per GPU (sharded by EP), so they do NOT
    need gradient sync. Only attention, embedding, layernorm, and router
    weights are replicated and need all-reduce.
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if ".experts." in name:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)


# ================================================================
# Training Loop
# ================================================================


@torch.no_grad()
def evaluate(model, val_loader, config, device):
    model.eval()
    total_loss = 0.0
    total_aux = 0.0
    count = 0
    max_eval_batches = 20

    for i, (input_ids, labels) in enumerate(val_loader):
        if i >= max_eval_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits, aux_loss = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size), labels.view(-1)
        )
        total_loss += loss.item()
        total_aux += aux_loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1), total_aux / max(count, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Expert Parallelism -- hand-written all-to-all")
    p.add_argument("--config", type=str, default="mini", choices=list(MOE_CONFIGS.keys()))
    p.add_argument("--ep-size", type=int, default=None,
                   help="EP group size. Default = world_size (pure EP, no DP)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--data-dir", type=str, default="../")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--num-experts", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--aux-loss-weight", type=float, default=None)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    # --- EP + DP group setup ---
    ep_size = args.ep_size if args.ep_size is not None else world_size
    assert world_size % ep_size == 0, f"world_size ({world_size}) must be divisible by ep_size ({ep_size})"
    dp_size = world_size // ep_size

    # EP groups: ranks that share experts via all-to-all
    ep_start = (rank // ep_size) * ep_size
    ep_ranks = list(range(ep_start, ep_start + ep_size))
    ep_group = dist.new_group(ranks=ep_ranks)
    ep_rank = dist.get_rank(ep_group)

    # DP groups: ranks that see different data, all-reduce gradients
    dp_group = None
    dp_rank = 0
    if dp_size > 1:
        dp_ranks = [rank % ep_size + i * ep_size for i in range(dp_size)]
        dp_group = dist.new_group(ranks=dp_ranks)
        dp_rank = dist.get_rank(dp_group)

    config = MOE_CONFIGS[args.config]
    if args.num_experts is not None:
        config.num_experts = args.num_experts
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.aux_loss_weight is not None:
        config.aux_loss_weight = args.aux_loss_weight

    assert config.num_experts % ep_size == 0, \
        f"num_experts ({config.num_experts}) must be divisible by ep_size ({ep_size})"

    seq_len = min(args.seq_len, config.max_seq_len)
    experts_per_rank = config.num_experts // ep_size

    if rank == 0:
        logger.info("=" * 60)
        logger.info("  Expert Parallelism -- GPT-2 %s", args.config)
        logger.info("=" * 60)
        logger.info("world_size=%d  ep_size=%d  dp_size=%d", world_size, ep_size, dp_size)
        logger.info("num_experts=%d  experts_per_rank=%d  top_k=%d",
                     config.num_experts, experts_per_rank, config.top_k)
        logger.info(
            "d_model=%d  n_heads=%d  d_ff=%d  n_layers=%d",
            config.d_model, config.n_heads, config.d_ff, config.n_layers,
        )

    # --- Model (each GPU only holds local experts) ---
    model = MoEGPT(config, ep_size, ep_group).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_params = count_parameters(model)
    mem_model = get_gpu_memory_mb(device)

    if rank == 0:
        logger.info("Params per GPU: %s", f"{total_params:,}")
        logger.info("Model memory:   %.1f MB", mem_model)

    # --- Data ---
    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "validation.bin")
    train_loader, train_sampler = create_dataloader(
        train_path, seq_len, args.batch_size,
        dp_rank=dp_rank, dp_size=dp_size,
    )
    val_loader, _ = create_dataloader(
        val_path, seq_len, args.batch_size,
        dp_rank=dp_rank, dp_size=dp_size, shuffle=False,
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

        logits, aux_loss = model(input_ids)                         # (B, T, V)
        lm_loss = F.cross_entropy(
            logits.view(-1, config.vocab_size), labels.view(-1)
        )
        total_loss = lm_loss + config.aux_loss_weight * aux_loss

        total_loss.backward()

        if dp_group is not None:
            sync_dp_gradients(model, dp_group, ep_size)

        optimizer.step()

        total_tokens += input_ids.numel()
        tokens_per_sec = total_tokens / (time.perf_counter() - t_start)

        step_record = {
            "step": step,
            "train_loss": round(lm_loss.item(), 4),
            "aux_loss": round(aux_loss.item(), 4),
            "total_loss": round(total_loss.item(), 4),
        }
        history.append(step_record)

        if rank == 0 and step % 10 == 0:
            logger.info(
                "step %4d/%d  lm=%.4f  aux=%.4f  total=%.4f  tok/s=%.0f",
                step, args.num_steps, lm_loss.item(),
                aux_loss.item(), total_loss.item(), tokens_per_sec,
            )

        if step % args.eval_interval == 0 or step == args.num_steps:
            val_loss, val_aux = evaluate(model, val_loader, config, device)
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
            "mode": "expert_parallel",
            "config": args.config,
            "world_size": world_size,
            "ep_size": ep_size,
            "dp_size": dp_size,
            "num_experts": config.num_experts,
            "experts_per_rank": experts_per_rank,
            "top_k": config.top_k,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "batch_size": args.batch_size,
            "seq_len": seq_len,
            "num_steps": args.num_steps,
            "params_per_gpu": total_params,
            "mem_model_mb": round(mem_model, 2),
            "mem_peak_mb": round(peak, 2),
            "elapsed_sec": round(elapsed, 2),
            "tokens_per_sec": round(total_tokens / elapsed, 1),
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1].get("val_loss"),
            "history": history,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(
            args.output_dir,
            f"results_train_gpt_ep_{args.config}_ep{ep_size}_dp{dp_size}.json",
        )
        with open(out_path, "w") as fout:
            json.dump(results, fout, indent=2)

        logger.info("=" * 60)
        logger.info("  EP GPT-2 %s -- ep=%d dp=%d", args.config.upper(), ep_size, dp_size)
        logger.info("=" * 60)
        logger.info("  %-25s %12s", "Params Per GPU", f"{total_params:,}")
        logger.info("  %-25s %12d", "Experts Per GPU", experts_per_rank)
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
