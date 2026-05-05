"""GPT-2 with Expert Parallelism -- DTensor-based using ExpertParallel.

Replaces the hand-written all-to-all dispatch from Part 8 with PyTorch's
DTensor ExpertParallel ParallelStyle:
  - 2D DeviceMesh with (dp, ep) dimensions
  - Expert weights stored as 3D tensors (num_experts, hidden, dim) in GroupedExperts
  - ExpertParallel shards weights via Shard(0) on the EP mesh dimension
  - AllToAllTokenDispatcher handles permute -> all-to-all -> compute -> combine

The model code is identical to single-GPU -- only the parallelization wrapper
changes. This is the key DTensor payoff.

Example:
    # Pure EP: 4 GPUs, all in one EP group
    torchrun --nproc_per_node=4 src/train_gpt_ep_dtensor.py --config mini --ep-size 4

    # EP+DP: 4 GPUs, ep_size=2, dp_size=2
    torchrun --nproc_per_node=4 src/train_gpt_ep_dtensor.py --config mini --ep-size 2
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
from torch.distributed._functional_collectives import all_to_all_single as functional_a2a
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle
from utils import count_parameters, get_gpu_memory_mb, get_gpu_peak_memory_mb

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


MOE_CONFIGS = {
    "mini": MoEGPTConfig(
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        vocab_size=50_257,
        max_seq_len=512,
        dropout=0.1,
        num_experts=8,
        top_k=2,
        aux_loss_weight=0.01,
    ),
    "small": MoEGPTConfig(
        d_model=768,
        n_heads=12,
        d_ff=3072,
        n_layers=12,
        vocab_size=50_257,
        max_seq_len=1024,
        dropout=0.1,
        num_experts=8,
        top_k=2,
        aux_loss_weight=0.01,
    ),
    "medium": MoEGPTConfig(
        d_model=1024,
        n_heads=16,
        d_ff=4096,
        n_layers=24,
        vocab_size=50_257,
        max_seq_len=1024,
        dropout=0.1,
        num_experts=8,
        top_k=2,
        aux_loss_weight=0.01,
    ),
}


# ================================================================
# Multi-Head Attention (same as Parts 7-8)
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


# ================================================================
# Router
# ================================================================


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        router_logits = self.gate(x)  # (B, T, num_experts)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (B, T, top_k) each
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B, T, top_k)
        return top_k_weights, top_k_indices, router_logits


# ================================================================
# Token Dispatcher (handles all-to-all for EP)
# ================================================================


class AllToAllTokenDispatcher:
    """Handles token permutation, all-to-all dispatch, and combine for EP.

    This is the DTensor-aware dispatcher: ExpertParallel sets ep_group
    on this object during distribute_module.
    """

    def __init__(self, num_experts: int, top_k: int):
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_group: dist.ProcessGroup | None = None

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Permute tokens by expert, all-to-all to expert-owning GPU.

        Args:
            x: (N, d_model) flattened input tokens
            top_scores: (N, top_k) routing weights
            expert_indices: (N, top_k) expert indices

        Returns:
            recv_tokens: tokens for this GPU's local experts
            num_tokens_per_local_expert: count per local expert
            metadata: dict for combine()
        """
        N, C = x.shape
        ep_size = dist.get_world_size(self.ep_group) if self.ep_group else 1

        scores_flat = top_scores.view(-1)  # (N*top_k,)
        indices_flat = expert_indices.view(-1)  # (N*top_k,)

        sort_order = torch.argsort(indices_flat, stable=True)  # (N*top_k,)
        token_source = sort_order // self.top_k  # original token index
        scores_sorted = scores_flat[sort_order]  # (N*top_k,)

        x_routed = x[token_source]  # (N*top_k, d_model)
        x_routed = x_routed * scores_sorted.unsqueeze(-1)  # apply routing weight

        if ep_size <= 1:
            num_per_expert = torch.histc(
                indices_flat.float(),
                bins=self.num_experts,
                min=0,
                max=self.num_experts,
            ).int()
            metadata = {
                "token_source": token_source,
                "scores_sorted": scores_sorted,
                "sort_order": sort_order,
                "N": N,
                "C": C,
                "input_splits": None,
                "output_splits": None,
                "ep_size": 1,
            }
            return x_routed, num_per_expert, metadata

        experts_per_rank = self.num_experts // ep_size
        sorted_indices = indices_flat[sort_order]
        target_rank = sorted_indices // experts_per_rank

        input_splits = [(target_rank == r).sum().item() for r in range(ep_size)]
        input_splits_t = torch.tensor(input_splits, dtype=torch.long, device=x.device)
        output_splits_t = torch.empty_like(input_splits_t)
        dist.all_to_all_single(output_splits_t, input_splits_t, group=self.ep_group)
        output_splits = output_splits_t.tolist()

        total_recv = sum(output_splits)
        recv_tokens = torch.empty(total_recv, C, dtype=x.dtype, device=x.device)
        dist.all_to_all_single(recv_tokens, x_routed, output_splits, input_splits, group=self.ep_group)

        recv_indices = torch.empty(total_recv, dtype=sorted_indices.dtype, device=x.device)
        dist.all_to_all_single(recv_indices, sorted_indices, output_splits, input_splits, group=self.ep_group)
        ep_rank = dist.get_rank(self.ep_group)
        local_indices = recv_indices - ep_rank * experts_per_rank

        num_per_expert = torch.histc(
            local_indices.float(),
            bins=experts_per_rank,
            min=0,
            max=experts_per_rank,
        ).int()

        metadata = {
            "token_source": token_source,
            "scores_sorted": scores_sorted,
            "sort_order": sort_order,
            "N": N,
            "C": C,
            "input_splits": input_splits,
            "output_splits": output_splits,
            "ep_size": ep_size,
        }
        return recv_tokens, num_per_expert, metadata

    def combine(self, expert_output: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Reverse the dispatch: all-to-all return, then scatter_add to original positions."""
        N = metadata["N"]
        C = metadata["C"]
        ep_size = metadata["ep_size"]
        token_source = metadata["token_source"]

        if ep_size > 1:
            in_splits = metadata["input_splits"]
            out_splits = metadata["output_splits"]
            total_orig = sum(in_splits)
            recv_back = torch.empty(total_orig, C, dtype=expert_output.dtype, device=expert_output.device)
            dist.all_to_all_single(recv_back, expert_output, in_splits, out_splits, group=self.ep_group)
        else:
            recv_back = expert_output

        output = torch.zeros(N, C, dtype=recv_back.dtype, device=recv_back.device)
        output.scatter_add_(0, token_source.unsqueeze(-1).expand(-1, C), recv_back)
        return output  # (N, d_model)


# ================================================================
# Grouped Experts (3D weight tensors for DTensor sharding)
# ================================================================


class GroupedExperts(nn.Module):
    """Expert FFNs stored as 3D weight tensors instead of nn.ModuleList.

    Weight shapes:
      w1: (num_experts, d_ff, d_model)   -- up projection
      w2: (num_experts, d_model, d_ff)   -- down projection

    ExpertParallel shards these on dim 0 (expert dimension) via Shard(0),
    so each GPU gets (num_experts / ep_size, ...) slices.

    The token_dispatcher attribute is required by ExpertParallel -- it sets
    ep_group on it during distribute_module.

    IMPORTANT: Router lives OUTSIDE this module (in MoETransformerBlock) so
    that distribute_module only touches expert weights and the dispatcher,
    not the router's 2D gate weight.
    """

    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.top_k = config.top_k

        self.w1 = nn.Parameter(torch.empty(config.num_experts, config.d_ff, config.d_model))  # (E, d_ff, d_model)
        self.w2 = nn.Parameter(torch.empty(config.num_experts, config.d_model, config.d_ff))  # (E, d_model, d_ff)

        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.token_dispatcher = AllToAllTokenDispatcher(
            config.num_experts,
            config.top_k,
        )

    def _experts_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Run local experts on pre-dispatched tokens."""
        from torch.distributed.tensor import DTensor

        w1 = self.w1.to_local() if isinstance(self.w1, DTensor) else self.w1
        w2 = self.w2.to_local() if isinstance(self.w2, DTensor) else self.w2

        num_per_expert_list = num_tokens_per_expert.tolist()
        x_splits = torch.split(x, num_per_expert_list, dim=0)

        outputs = []
        for i, x_expert in enumerate(x_splits):
            if x_expert.shape[0] == 0:
                outputs.append(x_expert)
                continue
            h = F.gelu(x_expert @ w1[i].T)  # (n_i, d_ff)
            h = h @ w2[i].T  # (n_i, d_model)
            outputs.append(h)

        return torch.cat(outputs, dim=0) if outputs else x[:0]

    def forward(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, and combine results.

        Args:
            x: (B, T, d_model) input hidden states
            routing_weights: (B, T, top_k) softmax weights from router
            selected_experts: (B, T, top_k) expert indices from router
        """
        B, T, C = x.shape  # (B, T, d_model)
        x_flat = x.view(-1, C)  # (B*T, d_model)

        recv_tokens, num_per_expert, metadata = self.token_dispatcher.dispatch(
            x_flat,
            routing_weights.view(-1, self.top_k),
            selected_experts.view(-1, self.top_k),
        )

        expert_output = self._experts_forward(recv_tokens, num_per_expert)

        combined = self.token_dispatcher.combine(expert_output, metadata)
        return combined.view(B, T, C)  # (B, T, d_model)


# ================================================================
# ExpertParallel ParallelStyle
# ================================================================


class ExpertParallel(ParallelStyle):
    """Shard expert weights on dim 0 and set ep_group on the dispatcher.

    This is the DTensor equivalent of the hand-written EP group setup from
    Part 8. A single distribute_module call replaces ~20 lines of manual
    process group creation and parameter partitioning.
    """

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        # Only shard 3D expert weight tensors (num_experts, ..., ...),
        # skip 2D params like the router gate weight.
        for param_name, param in mod.named_parameters(recurse=False):
            if param.dim() >= 3:
                dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
                mod.register_parameter(param_name, dist_param)

        if hasattr(mod, "token_dispatcher"):
            mod.token_dispatcher.ep_group = device_mesh.get_group()

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(module, device_mesh, partition_fn=self._partition_fn)


# ================================================================
# MoE Transformer Block (DTensor version)
# ================================================================


class MoETransformerBlock(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.router = Router(config.d_model, config.num_experts, config.top_k)
        self.moe = GroupedExperts(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))  # (B, T, d_model)
        h = self.ln2(x)  # (B, T, d_model)
        routing_weights, selected_experts, router_logits = self.router(h)
        moe_out = self.moe(h, routing_weights, selected_experts)  # (B, T, d_model)
        x = x + moe_out

        aux_loss = self._load_balancing_loss(router_logits, selected_experts)
        return x, aux_loss

    def _load_balancing_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = router_logits.shape
        num_tokens = B * T
        flat_experts = selected_experts.view(-1, self.top_k)
        expert_mask = torch.zeros(num_tokens, self.num_experts, device=router_logits.device)
        expert_mask.scatter_(1, flat_experts, 1.0)
        f = expert_mask.mean(dim=0)  # (num_experts,)
        P = F.softmax(router_logits, dim=-1).mean(dim=(0, 1))  # (num_experts,)
        return self.num_experts * (f * P).sum()


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
        self.blocks = nn.ModuleList([MoETransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.emb_dropout(self.tok_emb(input_ids) + self.pos_emb(pos))

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss
        total_aux_loss = total_aux_loss / self.config.n_layers

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits, total_aux_loss


# ================================================================
# Apply Expert Parallelism
# ================================================================


def apply_expert_parallelism(model: MoEGPT, ep_mesh: DeviceMesh) -> MoEGPT:
    """Apply ExpertParallel to all GroupedExperts modules in the model."""
    ep_style = ExpertParallel()
    for block in model.blocks:
        ep_style._apply(block.moe, ep_mesh)
    return model


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
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        total_loss += loss.item()
        total_aux += aux_loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1), total_aux / max(count, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Expert Parallelism -- DTensor ExpertParallel")
    p.add_argument("--config", type=str, default="mini", choices=list(MOE_CONFIGS.keys()))
    p.add_argument("--ep-size", type=int, default=None, help="EP group size. Default = world_size (pure EP, no DP)")
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
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    ep_size = args.ep_size if args.ep_size is not None else world_size
    assert world_size % ep_size == 0
    dp_size = world_size // ep_size

    config = MOE_CONFIGS[args.config]
    if args.num_experts is not None:
        config.num_experts = args.num_experts
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.aux_loss_weight is not None:
        config.aux_loss_weight = args.aux_loss_weight

    assert config.num_experts % ep_size == 0
    seq_len = min(args.seq_len, config.max_seq_len)
    experts_per_rank = config.num_experts // ep_size

    # --- 2D DeviceMesh: (dp, ep) ---
    if dp_size > 1:
        mesh_2d = []
        for dp_idx in range(dp_size):
            ep_ranks = [dp_idx * ep_size + i for i in range(ep_size)]
            mesh_2d.append(ep_ranks)
        mesh = DeviceMesh("cuda", mesh_2d, mesh_dim_names=("dp", "ep"))
        ep_mesh = mesh["ep"]
        dp_mesh = mesh["dp"]
    else:
        mesh = DeviceMesh("cuda", list(range(ep_size)), mesh_dim_names=("ep",))
        ep_mesh = mesh
        dp_mesh = None

    dp_rank = 0 if dp_mesh is None else dp_mesh.get_local_rank()

    if rank == 0:
        logger.info("=" * 60)
        logger.info("  DTensor Expert Parallelism -- GPT-2 %s", args.config)
        logger.info("=" * 60)
        logger.info("world_size=%d  ep_size=%d  dp_size=%d", world_size, ep_size, dp_size)
        logger.info("DeviceMesh: %s", mesh)
        logger.info("num_experts=%d  experts_per_rank=%d  top_k=%d", config.num_experts, experts_per_rank, config.top_k)

    # --- Model ---
    model = MoEGPT(config).to(device)

    # Apply ExpertParallel: shards expert weights on dim 0, sets ep_group
    apply_expert_parallelism(model, ep_mesh)

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
        train_path,
        seq_len,
        args.batch_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )
    val_loader, _ = create_dataloader(
        val_path,
        seq_len,
        args.batch_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        shuffle=False,
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

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, aux_loss = model(input_ids)
        lm_loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        total_loss = lm_loss + config.aux_loss_weight * aux_loss

        total_loss.backward()

        # For DP: all-reduce non-expert gradients across dp_mesh
        if dp_mesh is not None:
            dp_group = dp_mesh.get_group()
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                if ".w1" in name or ".w2" in name:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)

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
                step,
                args.num_steps,
                lm_loss.item(),
                aux_loss.item(),
                total_loss.item(),
                tokens_per_sec,
            )

        if step % args.eval_interval == 0 or step == args.num_steps:
            val_loss, val_aux = evaluate(model, val_loader, config, device)
            if rank == 0:
                logger.info(
                    "  [eval] step %d  val_loss=%.4f  val_aux=%.4f",
                    step,
                    val_loss,
                    val_aux,
                )
            step_record["val_loss"] = round(val_loss, 4)
            step_record["val_aux_loss"] = round(val_aux, 4)

    # --- Results ---
    peak = get_gpu_peak_memory_mb(device)
    elapsed = time.perf_counter() - t_start

    if rank == 0:
        results = {
            "mode": "expert_parallel_dtensor",
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
            f"results_train_gpt_ep_dtensor_{args.config}_ep{ep_size}_dp{dp_size}.json",
        )
        with open(out_path, "w") as fout:
            json.dump(results, fout, indent=2)

        logger.info("=" * 60)
        logger.info("  DTensor EP GPT-2 %s -- ep=%d dp=%d", args.config.upper(), ep_size, dp_size)
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
