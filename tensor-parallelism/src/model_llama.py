"""Standard (non-parallelized) Llama-style transformer model.

A minimal Llama decoder used as the baseline for tensor-parallelism experiments.
Differences from the GPT model (model.py):
  1. RMSNorm instead of LayerNorm
  2. Rotary Position Embeddings (RoPE) instead of learned positional embeddings
  3. Grouped Query Attention (GQA) instead of standard multi-head attention
  4. SwiGLU FFN instead of GELU FFN (3 weight matrices instead of 2)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_parameters, get_gpu_memory_mb, get_gpu_peak_memory_mb


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

D_MODEL = 512       # hidden dimension (embedding size)
N_HEADS = 8         # number of attention heads
D_HEAD = D_MODEL // N_HEADS   # = 64, dimension per head
D_FF = 2048         # feed-forward intermediate dimension (4x D_MODEL)
N_LAYERS = 6        # number of transformer blocks
VOCAB_SIZE = 10_000
MAX_SEQ_LEN = 512

N_KV_HEADS = 4                   # GQA: fewer KV heads than Q heads
D_KV = N_KV_HEADS * D_HEAD       # = 256, total KV dimension
GQA_GROUP_SIZE = N_HEADS // N_KV_HEADS  # = 2, Q heads per KV head

# SwiGLU uses 2/3 of the typical 4x expansion to keep param count similar,
# since it has 3 matrices instead of 2.  floor to multiple of 256.
D_FF_LLAMA = (int(D_MODEL * 8 / 3) // 256) * 256  # = 1024

BATCH_SIZE = 8
SEQ_LEN = 256
NUM_WARMUP = 3
NUM_BENCHMARK = 10


# ---------------------------------------------------------------------------
# RMSNorm (replaces LayerNorm)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm skips the mean-centering step and only
    normalizes by the root-mean-square, then scales by a learned gamma.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute the complex-valued rotation frequencies for RoPE.

    Returns a [max_seq_len, d_head/2] complex tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to Q or K tensor.

    x: [B, n_heads, T, d_head]
    freqs: [T, d_head/2] complex
    """
    # Pair up consecutive dimensions as complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x_complex.shape[-2], :].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


# ---------------------------------------------------------------------------
# Grouped Query Attention (replaces standard MHA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA).

    Uses N_HEADS Q heads but only N_KV_HEADS K/V heads. Each KV head is
    shared by GQA_GROUP_SIZE Q heads, reducing KV cache and compute.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.d_head, max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # RoPE applied to Q and K only (not V) - see diagram
        Q = apply_rope(Q, self.rope_freqs)
        K = apply_rope(K, self.rope_freqs)

        # Expand KV heads to match Q heads:  [B, n_kv, T, d] -> [B, n_q, T, d]
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network (replaces GELU FFN)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: gate + up projections with SiLU gating, then down projection.

    output = W_down( SiLU(W_gate(x)) * W_up(x) )

    3 weight matrices instead of 2, so D_FF is reduced to ~2/3 of the
    GPT 4x expansion to keep total parameters comparable.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


# ---------------------------------------------------------------------------
# Llama Transformer Block
# ---------------------------------------------------------------------------

class LlamaTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, max_seq_len: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full Llama Model
# ---------------------------------------------------------------------------

class StandardLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        # No positional embedding - RoPE is applied inside attention
        self.blocks = nn.ModuleList([
            LlamaTransformerBlock(D_MODEL, N_HEADS, N_KV_HEADS,
                                  D_FF_LLAMA, MAX_SEQ_LEN)
            for _ in range(N_LAYERS)
        ])
        self.norm_f = RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)
