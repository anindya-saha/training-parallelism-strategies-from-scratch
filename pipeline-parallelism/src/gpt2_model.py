"""Small GPT for pipeline-parallel tutorials (same math as tensor-parallelism ``model_gpt``).

**Why this layout matches ``model.py`` (ToyModel)**

In production you still have a **single** ``nn.Module`` (class + checkpoint). Nobody
hands you a Python list of layers in isolation; the list is **derived** from that
module. Here ``StandardGPT`` keeps every pipeline chunk in one ``nn.ModuleList`` called
``layers`` (embedding, blocks, final norm, lm head), same idea as ``ToyModel.layers``.
``get_stage()`` builds the full model on the ``meta`` device, does
``list(full_model.layers)``, slices by rank, then materializes only the local
``nn.Sequential`` on the real GPU -- the same pattern as ``model.get_stage``.

DeepSpeed ``PipelineModule`` and similar APIs often take an explicit ``layers=`` list;
that list is normally produced by **walking** a model you already have (or the model
is authored with a ``ModuleList`` field exactly so partitioning can read it).

Split rule: ``ceil(n / world_size)`` modules per rank (layer-count chunking for learning).
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Defaults: small enough for 2-GPU learning runs
DEFAULT_D_MODEL = 256
DEFAULT_N_HEADS = 4
DEFAULT_D_FF = 1024
DEFAULT_N_LAYERS = 4
DEFAULT_VOCAB_SIZE = 1000
DEFAULT_MAX_SEQ_LEN = 128


class StandardAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.W_q(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.W_o(out)


class StandardFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=bias)
        self.W2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.gelu(self.W1(x)))


class StandardTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_bias: bool = False,
        ffn_bias: bool = True,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardAttention(d_model, n_heads, bias=attn_bias)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = StandardFFN(d_model, d_ff, bias=ffn_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TokPosEmbedding(nn.Module):
    """Maps ``input_ids`` (B, T) int64 to hidden states (B, T, d_model)."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        return self.tok_emb(input_ids) + self.pos_emb(pos)


class StandardGPT(nn.Module):
    """One module, ordered ``layers`` list -- partition by slicing ``layers`` (see ``get_stage``)."""

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
        self.max_seq_len = max_seq_len
        parts: list[nn.Module] = [
            TokPosEmbedding(vocab_size, d_model, max_seq_len),
        ]
        for _ in range(n_layers):
            parts.append(
                StandardTransformerBlock(
                    d_model, n_heads, d_ff, attn_bias=attn_bias, ffn_bias=ffn_bias
                )
            )
        parts.append(nn.LayerNorm(d_model))
        parts.append(nn.Linear(d_model, vocab_size, bias=False))
        self.layers = nn.ModuleList(parts)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](input_ids)
        for i in range(1, len(self.layers)):
            x = self.layers[i](x)
        return x


def log_model_info(model: nn.Module, label: str = "Model") -> None:
    logger.info("%s repr:\n%s", label, model)
    total_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        logger.info(
            "  %s: %r shape=%s dtype=%s (%d params)",
            label,
            name,
            tuple(param.shape),
            param.dtype,
            n,
        )
    logger.info("%s total parameters: %d", label, total_params)


def get_stage(
    rank: int,
    world_size: int,
    device: torch.device,
    d_model: int = DEFAULT_D_MODEL,
    n_heads: int = DEFAULT_N_HEADS,
    d_ff: int = DEFAULT_D_FF,
    n_layers: int = DEFAULT_N_LAYERS,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    attn_bias: bool = False,
    ffn_bias: bool = True,
    log_full: bool = True,
    log_stage: bool = True,
    quiet: bool = False,
) -> tuple[nn.Sequential, int, int]:
    """Return ``(stage, start_idx, end_idx)`` for this rank's contiguous slice of ``StandardGPT.layers``.

    Same flow as ``model.get_stage`` for ``ToyModel``: build the **full** model on
    ``meta``, read ``list(full_model.layers)``, slice by index, materialize local
    ``nn.Sequential`` on ``device``.

    First stage may start with ``TokPosEmbedding`` (expects ``input_ids``); later stages
    expect activations ``(B, T, d_model)`` float.

    Split rule: ``ceil(n / world_size)`` modules per rank (same as ``model.get_stage``).
    """
    with torch.device("meta"):
        full_model = StandardGPT(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            attn_bias=attn_bias,
            ffn_bias=ffn_bias,
        )

    if not quiet and rank == 0 and log_full:
        log_model_info(full_model, label="Full StandardGPT (meta)")

    all_layers = list(full_model.layers)
    n_mod = len(all_layers)
    chunk = (n_mod + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n_mod)
    local: Sequence[nn.Module] = all_layers[start:end]

    if not quiet:
        logger.info(
            "Rank %d: stage owns modules [%d:%d] of %d (from full_model.layers)",
            rank,
            start,
            end,
            n_mod,
        )

    stage = nn.Sequential(*local)
    stage = stage.to_empty(device=device)
    stage.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )

    if not quiet and log_stage:
        log_model_info(stage, label=f"Rank {rank} stage")
    return stage, start, end


def load_stage_from_full_state_dict(
    stage: nn.Sequential,
    full_sd: dict[str, torch.Tensor],
    global_start: int,
    device: torch.device,
) -> None:
    """Copy weights from a ``StandardGPT`` ``state_dict`` into a pipeline ``Sequential``.

    Full keys look like ``layers.{j}....``; stage keys look like ``{local_i}....`` where
    ``global_start + local_i == j``.
    """
    stage_sd: dict[str, torch.Tensor] = {}
    for li in range(len(stage)):
        gj = global_start + li
        prefix_full = f"layers.{gj}."
        prefix_stage = f"{li}."
        for k, v in full_sd.items():
            if k.startswith(prefix_full):
                suffix = k[len(prefix_full) :]
                stage_sd[prefix_stage + suffix] = v.to(device=device)
    stage.load_state_dict(stage_sd, strict=True)


def merge_pipeline_stage_state_dicts_to_full(
    stage_sds: list[tuple[dict[str, torch.Tensor], int]],
) -> dict[str, torch.Tensor]:
    """Merge per-rank ``Sequential`` state dicts into one ``StandardGPT`` state dict."""
    full_sd: dict[str, torch.Tensor] = {}
    for sd, global_start in stage_sds:
        for k, v in sd.items():
            parts = k.split(".", 1)
            local_i = int(parts[0])
            rest = parts[1]
            gj = global_start + local_i
            full_sd[f"layers.{gj}.{rest}"] = v
    return full_sd


def lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Causal LM loss: predict next token (same mask as single-GPT training)."""
    vocab = logits.size(-1)
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, vocab),
        input_ids[:, 1:].contiguous().view(-1),
    )


def pipeline_model_config_string(
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_layers: int,
    vocab_size: int,
    max_seq_len: int,
) -> str:
    return (
        f"d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, n_layers={n_layers}, "
        f"vocab={vocab_size}, max_seq_len={max_seq_len}"
    )
