"""
GPT-2 Style Transformer Model for Pipeline Parallelism
=======================================================
Each layer is a standalone nn.Module that takes a single tensor
and returns a single tensor — required for DeepSpeed PipelineModule.

Model: 48 layers, hidden_dim=1536, 24 heads, vocab=32000
Total parameters: ~1.5B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model hyperparameters ────────────────────────────────────
NUM_LAYERS   = 96
HIDDEN_DIM   = 1536
NUM_HEADS    = 24
VOCAB_SIZE   = 32000
MAX_SEQ_LEN  = 2048
DROPOUT      = 0.0       # Disable for benchmarking


class EmbeddingLayer(nn.Module):
    """Token + positional embeddings. First layer of the pipeline."""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, HIDDEN_DIM)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, input_ids):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        return self.drop(x)


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block (GPT-2 style)."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(HIDDEN_DIM)
        self.attn = nn.MultiheadAttention(
            HIDDEN_DIM, NUM_HEADS, dropout=DROPOUT, batch_first=True
        )
        self.ln2 = nn.LayerNorm(HIDDEN_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 4 * HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(4 * HIDDEN_DIM, HIDDEN_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        S = x.size(1)
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=causal_mask, is_causal=True)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class LMHead(nn.Module):
    """Final layer norm + linear projection to vocab. Last pipeline stage."""

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(HIDDEN_DIM)
        self.proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

    def forward(self, x):
        return self.proj(self.ln(x))


def get_layers():
    """Return the model as a flat list of nn.Module layers for PipelineModule."""
    layers = [EmbeddingLayer()]
    for _ in range(NUM_LAYERS):
        layers.append(TransformerBlock())
    layers.append(LMHead())
    return layers


def get_sequential_model():
    """Return the model as nn.Sequential (for single-GPU baseline)."""
    return nn.Sequential(*get_layers())


def lm_loss_fn(logits, labels):
    """Cross-entropy loss for language modeling. Used by DeepSpeed pipeline."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = get_sequential_model()
    n_params = count_parameters(model)
    print(f"Model: {NUM_LAYERS} layers, hidden={HIDDEN_DIM}, heads={NUM_HEADS}")
    print(f"Parameters: {n_params:,} ({n_params/1e9:.2f}B)")
    print(f"FP16 weight size: {n_params * 2 / 1e9:.2f} GB")
    print(f"Adam optimizer states (FP32): {n_params * 12 / 1e9:.2f} GB")
    print(f"\nLayers: {len(get_layers())} (1 embedding + {NUM_LAYERS} transformer + 1 lm_head)")
