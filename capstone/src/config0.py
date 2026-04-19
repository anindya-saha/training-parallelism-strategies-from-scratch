"""
Configuration dataclasses for Vizz SLIM.
"""
from dataclasses import dataclass

@dataclass
class ModelConfig:
    block_size: int = 1024       # max sequence length
    vocab_size: int = 50304      # padded for efficiency (50257 -> 50304, divisible by 128)
    n_layer: int = 12            # number of transformer layers
    n_head: int = 12             # number of attention heads
    n_embd: int = 768            # embedding dimension
    # MoE settings (only active when num_experts > 0)
    num_experts: int = 0         # 0 = dense model, >0 = MoE
    moe_top_k: int = 2           # number of experts selected per token
    moe_freq: int = 2            # apply MoE every moe_freq layers (1=all, 2=every other)
    aux_loss_coeff: float = 0.01 # load balancing auxiliary loss coefficient

@dataclass
class ParallelConfig:
    tp_size: int = 1   # Tensor Parallelism degree
    pp_size: int = 1   # Pipeline Parallelism degree
    cp_size: int = 1   # Context Parallelism degree
    ep_size: int = 1   # Expert Parallelism degree
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
    micro_batch_size: int = 64      # micro batch size per GPU
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 715
    max_steps: int = 19073          # ~1 epoch for 10B tokens
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 250
    checkpoint_interval: int = 5000
    data_root: str = "edu_fineweb10B"
