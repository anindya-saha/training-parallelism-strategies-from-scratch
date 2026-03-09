"""Standard (non-parallelized) GPT-style transformer model.

A minimal GPT decoder used as the baseline for tensor-parallelism experiments.
All TP scripts import the model definition and shared constants from here.

Supports both Multi-Head Attention (MHA) and Grouped Query Attention (GQA).
Use ModelConfig to switch between presets or define custom model sizes.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Transformer model configuration.

    n_kv_heads controls attention type:
      n_kv_heads == n_heads   -> MHA (Multi-Head Attention)
      n_kv_heads == 1         -> MQA (Multi-Query Attention)
      1 < n_kv_heads < n_heads -> GQA (Grouped Query Attention)
    """

    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 6
    vocab_size: int = 10_000
    max_seq_len: int = 512

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """Number of times to repeat each KV head to match query heads."""
        return self.n_heads // self.n_kv_heads

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )


SMALL_CONFIG = ModelConfig()
LARGE_CONFIG = ModelConfig(
    d_model=2048,
    n_heads=16,
    n_kv_heads=4,
    d_ff=8192,
    n_layers=12,
)

# ---------------------------------------------------------------------------
# Backward-compatible constants (match SMALL_CONFIG)
# ---------------------------------------------------------------------------

D_MODEL = SMALL_CONFIG.d_model
N_HEADS = SMALL_CONFIG.n_heads
N_KV_HEADS = SMALL_CONFIG.n_kv_heads
D_HEAD = SMALL_CONFIG.d_head
D_FF = SMALL_CONFIG.d_ff
N_LAYERS = SMALL_CONFIG.n_layers
VOCAB_SIZE = SMALL_CONFIG.vocab_size
MAX_SEQ_LEN = SMALL_CONFIG.max_seq_len

BATCH_SIZE = 8
SEQ_LEN = 256
NUM_WARMUP = 3
NUM_BENCHMARK = 10


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA).

    (B, n_kv_heads, T, d_head) -> (B, n_heads, T, d_head)
    No-op when n_rep == 1 (standard MHA).
    """
    if n_rep == 1:
        return x
    B, n_kv, T, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv, n_rep, T, d)
        .reshape(B, n_kv * n_rep, T, d)
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_mb(device=0) -> float:
    return torch.cuda.memory_allocated(device) / 1024 / 1024


def get_gpu_peak_memory_mb(device=0) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class StandardAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


class StandardFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.gelu(self.W1(x)))


class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_kv_heads: Optional[int] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardAttention(d_model, n_heads, n_kv_heads=n_kv_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = StandardFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class StandardGPT(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = SMALL_CONFIG
        c = config
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_emb = nn.Embedding(c.max_seq_len, c.d_model)
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(
                c.d_model, c.n_heads, c.d_ff, n_kv_heads=c.n_kv_heads
            )
            for _ in range(c.n_layers)
        ])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
