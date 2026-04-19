"""
From-scratch training entry for Vizz SLIM.

All parallelism setup for this script lives here (DeviceMesh process groups).
Other modules are unchanged.

Usage:
  Single GPU:   python train.py
  DDP:          torchrun --standalone --nproc_per_node=4 capstone/src/train.py --data_root /mnt/fsx-static/asaha/parallelism-experiments/data/edu_fineweb10B
  5D mesh:      torchrun --standalone --nproc_per_node=4 capstone/src/train.py --data_root /mnt/fsx-static/asaha/parallelism-experiments/data/edu_fineweb10B --tp_size 2 --pp_size 2
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import logging
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# Fixed name so logs read "train" when this file is run as a script (__name__ == "__main__").
logger = logging.getLogger("train")


@dataclass
class ModelConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50304  # padded for efficiency (50257 -> 50304, divisible by 128)
    n_layer: int = 12  # number of transformer layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension
    # MoE settings (only active when num_experts > 0)
    num_experts: int = 0  # 0 = dense model, >0 = MoE
    moe_top_k: int = 2  # number of experts selected per token
    moe_freq: int = 2  # apply MoE every moe_freq layers (1=all, 2=every other)
    aux_loss_coeff: float = 0.01  # load balancing auxiliary loss coefficient


@dataclass
class ParallelConfig:
    tp_size: int = 1  # Tensor Parallelism degree
    pp_size: int = 1  # Pipeline Parallelism degree
    cp_size: int = 1  # Context Parallelism degree
    ep_size: int = 1  # Expert Parallelism degree
    dp_size: int = -1  # Data Parallelism degree (auto-computed)

    def validate(self, world_size: int):
        """Validate and compute dp_size."""
        model_parallel = self.tp_size * self.pp_size * self.cp_size * self.ep_size
        assert world_size % model_parallel == 0, (
            f"world_size ({world_size}) must be divisible by "
            f"TP({self.tp_size}) * PP({self.pp_size}) * CP({self.cp_size}) * EP({self.ep_size}) = {model_parallel}"
        )
        self.dp_size = world_size // model_parallel
        return self


@dataclass
class TrainConfig:
    total_batch_size: int = 524288  # ~0.5M tokens (2**19)
    micro_batch_size: int = 64  # micro batch size per data-parallel rank
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 715
    max_steps: int = 19073  # ~1 epoch for 10B tokens
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 250
    checkpoint_interval: int = 5000
    data_root: str = "edu_fineweb10B"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        device_type: str,
        master_process: bool = True,
    ) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        if master_process:
            num_decay = sum(p.numel() for p in decay_params)
            num_nodecay = sum(p.numel() for p in nodecay_params)
            logger.info(
                "optimizer: %d decayed param tensors (%s params), %d non-decayed (%s params)",
                len(decay_params), f"{num_decay:,}",
                len(nodecay_params), f"{num_nodecay:,}",
            )
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            logger.info("using fused AdamW: %s", use_fused)
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def dataclass_from_args(cls: type, args: argparse.Namespace):
    """Construct a dataclass from an argparse Namespace, using only matching fields."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in vars(args).items() if k in field_names})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5D parallelism training")
    # parallelism
    p.add_argument("--tp_size", type=int, default=1)
    p.add_argument("--pp_size", type=int, default=1)
    p.add_argument("--cp_size", type=int, default=1)
    p.add_argument("--ep_size", type=int, default=1)
    # model
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--num_experts", type=int, default=0)
    p.add_argument("--moe_top_k", type=int, default=2)
    # training
    p.add_argument("--total_batch_size", type=int, default=524288)
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--max_lr", type=float, default=6e-4)
    p.add_argument("--max_steps", type=int, default=19073)
    p.add_argument("--warmup_steps", type=int, default=715)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--data_root", type=str, default="edu_fineweb10B")
    return p.parse_args()


def setup_distributed() -> tuple[bool, int, int, int, bool, str]:
    """Returns (ddp, rank, local_rank, world_size, master_process, device_str)."""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA required for DDP"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master = rank == 0
        if master:
            logger.info(
                "Distributed env RANK=%s LOCAL_RANK=%s WORLD_SIZE=%s",
                rank,
                local_rank,
                world_size,
            )
        init_process_group(backend="nccl")
        device_str = f"cuda:{local_rank}"
        torch.cuda.set_device(device_str)
        if master:
            logger.info(
                "NCCL initialized local_rank=%s world_size=%s device=%s",
                local_rank,
                world_size,
                device_str,
            )
    else:
        rank = local_rank = 0
        world_size = 1
        master = True
        if torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"
        logger.info("Single-process mode rank=0 device=%s", device_str)
    return ddp, rank, local_rank, world_size, master, device_str


def create_process_groups_device_mesh(
    parallel_config: ParallelConfig,
) -> dict[str, dist.ProcessGroup | None]:
    """
    Build TP/PP/CP/EP/DP subgroups from a 5D DeviceMesh.

    Mesh shape (outer to inner): (dp, pp, cp, ep, tp). Last dim (tp) is fastest
    changing in row-major rank assignment.
    """
    assert dist.is_initialized(), "init_process_group must run first"
    world_size = dist.get_world_size()
    mesh_shape = (
        parallel_config.dp_size,
        parallel_config.pp_size,
        parallel_config.cp_size,
        parallel_config.ep_size,
        parallel_config.tp_size,
    )
    product = math.prod(mesh_shape)
    assert (
        product == world_size
    ), f"mesh product {product} must equal world_size {world_size}, got shape {mesh_shape}"

    mesh_dim_names = ("dp", "pp", "cp", "ep", "tp")
    mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=mesh_dim_names)
    if dist.get_rank() == 0:
        logger.info(
            "DeviceMesh shape=%s dim_names=%s",
            mesh_shape,
            mesh_dim_names,
        )

    groups: dict[str, dist.ProcessGroup | None] = {}
    for name, size in zip(mesh_dim_names, mesh_shape, strict=True):
        if size > 1:
            groups[name] = mesh.get_group(name)
        else:
            groups[name] = None
    return groups


def _data_parallel_rank_in_group(groups: dict[str, dist.ProcessGroup | None]) -> int:
    if groups.get("dp") is None:
        return 0
    return dist.get_rank(groups["dp"])


def _data_parallel_world_size(groups: dict[str, dist.ProcessGroup | None]) -> int:
    if groups.get("dp") is None:
        return 1
    return dist.get_world_size(groups["dp"])


def list_shard_paths(data_root: str, split: str) -> list[str]:
    """Sorted ``*.npy`` paths under ``{data_root}/{split}/`` (split dir must exist)."""
    split_dir = os.path.join(data_root, split)
    assert os.path.isdir(split_dir), f"missing split directory {split_dir!r}"
    names = [n for n in os.listdir(split_dir) if n.endswith(".npy")]
    names = sorted(names)
    paths = [os.path.join(split_dir, n) for n in names]
    assert len(paths) > 0, f"no .npy shards in {split_dir!r}"
    return paths


class FinewebTokenIterableDataset(IterableDataset):
    """
    Pre-tokenized .npy shards (same shard stride layout as data.DataLoaderLite).

    Shards are ``*.npy`` under ``{data_root}/{split}/`` (see ``list_shard_paths``).

    Each step yields one microbatch (x, y) of shape (B, T). Ranks stride by
    ``dp_world_size`` so data is disjoint across data-parallel replicas.

    Use ``DataLoader(..., batch_size=None)`` so each yield is one batch.

    If ``max_batches`` is set, ``__iter__`` stops after that many microbatches
    (used for validation like ``train.py`` val_loss_steps=20). If ``max_batches``
    is None, iteration runs forever (training).

    This iterator matches DataLoaderLite: ``num_workers`` must be 0 unless you
    add a worker-aware split inside ``__iter__``.
    """

    def __init__(
        self,
        micro_batch_size: int,
        block_size: int,
        split: str,
        data_root: str,
        dp_rank: int,
        dp_world_size: int,
        *,
        max_batches: int | None = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val")
        self.B = micro_batch_size
        self.T = block_size
        self.split = split
        self.data_root = data_root
        self.dp_rank = dp_rank
        self.dp_world = dp_world_size
        self.max_batches = max_batches

        self.shards = list_shard_paths(data_root, split)

    @staticmethod
    def load_tokens(filename: str) -> torch.Tensor:
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        return torch.tensor(npt, dtype=torch.long)

    def __iter__(self):
        wi = get_worker_info()
        if wi is not None and wi.num_workers > 0:
            raise RuntimeError(
                "FinewebTokenIterableDataset: set DataLoader num_workers=0 "
                "(or implement worker partitioning)."
            )

        B, T = self.B, self.T
        r, w = self.dp_rank, self.dp_world
        current_shard = 0
        tokens = self.load_tokens(self.shards[current_shard])
        pos = B * T * r
        batches_emitted = 0

        while True:
            if self.max_batches is not None and batches_emitted >= self.max_batches:
                return

            buf = tokens[pos : pos + B * T + 1]
            if len(buf) < B * T + 1:
                current_shard = (current_shard + 1) % len(self.shards)
                tokens = self.load_tokens(self.shards[current_shard])
                pos = B * T * r
                continue

            x = (buf[:-1]).view(B, T)
            y = (buf[1:]).view(B, T)
            pos += B * T * w
            batches_emitted += 1
            yield x, y

            if pos + (B * T * w + 1) > len(tokens):
                current_shard = (current_shard + 1) % len(self.shards)
                tokens = self.load_tokens(self.shards[current_shard])
                pos = B * T * r


def build_fineweb_dataloader(
    *,
    micro_batch_size: int,
    block_size: int,
    split: str,
    data_root: str,
    dp_rank: int,
    dp_world_size: int,
    pin_memory: bool,
    max_batches: int | None,
) -> DataLoader:
    ds = FinewebTokenIterableDataset(
        micro_batch_size,
        block_size,
        split,
        data_root,
        dp_rank,
        dp_world_size,
        max_batches=max_batches,
    )
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=0,
        pin_memory=pin_memory,
    )


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    ddp, rank, local_rank, world_size, master, device_str = setup_distributed()
    device = torch.device(device_str)
    device_type = "cuda" if device_str.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # -- parallelism config --------------------------------------------------
    par = ParallelConfig(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        cp_size=args.cp_size,
        ep_size=args.ep_size,
    )
    use_only_dp = (
        par.tp_size == 1 and par.pp_size == 1 and par.cp_size == 1 and par.ep_size == 1
    )

    if ddp:
        par.validate(world_size)
        if use_only_dp:
            groups: dict[str, dist.ProcessGroup | None] = {
                "tp": None, "pp": None, "cp": None, "ep": None, "dp": None,
            }
            dp_rank = rank
            dp_world = world_size
        else:
            groups = create_process_groups_device_mesh(par)
            dp_rank = _data_parallel_rank_in_group(groups)
            dp_world = _data_parallel_world_size(groups)
    else:
        groups = {"tp": None, "pp": None, "cp": None, "ep": None, "dp": None}
        dp_rank = 0
        dp_world = 1

    if master:
        if use_only_dp:
            logger.info("Parallelism: DDP only  dp_rank=%s dp_world=%s", dp_rank, dp_world)
        else:
            logger.info(
                "Parallelism: TP=%s PP=%s CP=%s EP=%s DP=%s  dp_rank=%s dp_world=%s",
                par.tp_size, par.pp_size, par.cp_size, par.ep_size, par.dp_size,
                dp_rank, dp_world,
            )

    # -- configs --------------------------------------------------------------
    m_cfg = dataclass_from_args(ModelConfig, args)
    t_cfg = dataclass_from_args(TrainConfig, args)

    B = t_cfg.micro_batch_size
    T = m_cfg.block_size
    assert t_cfg.total_batch_size % (B * T * dp_world) == 0, (
        f"total_batch_size ({t_cfg.total_batch_size}) must be divisible by "
        f"B*T*dp_world ({B}*{T}*{dp_world})"
    )
    grad_accum_steps = t_cfg.total_batch_size // (B * T * dp_world)
    if master:
        logger.info(
            "total_batch_size=%s  micro_batch_size=%s  block_size=%s  grad_accum_steps=%s",
            t_cfg.total_batch_size, B, T, grad_accum_steps,
        )

    # -- data -----------------------------------------------------------------
    pin_mem = device_type == "cuda"
    train_loader = build_fineweb_dataloader(
        micro_batch_size=B, block_size=T, split="train",
        data_root=t_cfg.data_root, dp_rank=dp_rank, dp_world_size=dp_world,
        pin_memory=pin_mem, max_batches=None,
    )
    val_loader = build_fineweb_dataloader(
        micro_batch_size=B, block_size=T, split="val",
        data_root=t_cfg.data_root, dp_rank=dp_rank, dp_world_size=dp_world,
        pin_memory=pin_mem, max_batches=20,
    )
    if master:
        logger.info(
            "DataLoader: train_shards=%s  val_shards=%s",
            len(train_loader.dataset.shards), len(val_loader.dataset.shards),
        )

    # -- model ----------------------------------------------------------------
    torch.set_float32_matmul_precision("high")
    model = GPT(m_cfg)
    model.to(device)
    raw_model = model

    if ddp and dp_world > 1:
        model = DDP(model, device_ids=[local_rank], process_group=groups["dp"])
        raw_model = model.module

    if master:
        n_params = sum(p.numel() for p in raw_model.parameters())
        logger.info("GPT  n_layer=%s n_head=%s n_embd=%s  params=%s",
                     m_cfg.n_layer, m_cfg.n_head, m_cfg.n_embd, f"{n_params:,}")

    # -- optimizer ------------------------------------------------------------
    optimizer = raw_model.configure_optimizers(
        weight_decay=t_cfg.weight_decay,
        learning_rate=t_cfg.max_lr,
        device_type=device_type,
        master_process=master,
    )

    # -- logging --------------------------------------------------------------
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    if master:
        with open(log_file, "w") as f:
            pass

    # -- training loop --------------------------------------------------------
    train_iter = iter(train_loader)
    for step in range(t_cfg.max_steps):
        t0 = time.time()
        last_step = step == t_cfg.max_steps - 1

        # -- validation -------------------------------------------------------
        if step % t_cfg.eval_interval == 0 or last_step:
            model.eval()
            val_iter = iter(val_loader)
            with torch.no_grad():
                val_loss_accum = torch.tensor(0.0, device=device)
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = next(val_iter)
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master:
                logger.info("step %5d | val_loss %.4f", step, val_loss_accum.item())
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # -- training step ----------------------------------------------------
        model.train()
        optimizer.zero_grad()
        loss_accum = torch.tensor(0.0, device=device)

        for micro_step in range(grad_accum_steps):
            x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            if ddp and dp_world > 1:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), t_cfg.grad_clip)

        lr = get_lr(step, t_cfg.warmup_steps, t_cfg.max_steps, t_cfg.max_lr, t_cfg.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = B * T * grad_accum_steps * dp_world
        tokens_per_sec = tokens_processed / dt

        if master and (step % t_cfg.log_interval == 0 or last_step):
            logger.info(
                "step %5d | loss %.6f | lr %.4e | norm %.4f | dt %.2fms | tok/s %.0f",
                step, loss_accum.item(), lr, norm, dt * 1000, tokens_per_sec,
            )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    # -- cleanup --------------------------------------------------------------
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
