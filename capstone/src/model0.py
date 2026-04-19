"""
GPT model architecture for Vizz SLIM.
Refactored from Karpathy's build-nanogpt/train_gpt2.py.
"""
import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import ModelConfig


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # set by apply_tensor_parallelism or apply_context_parallelism
        self.tp_group = None
        self.cp_group = None

    def forward(self, x):
        # Layout convention: LM blocks use (B, T, C) = batch, time, model width (n_embd).
        B, T, C = x.size()
        assert C == self.n_embd

        # Fused QKV linear: one matmul, then split into three (B, T, n_embd) tensors.
        qkv = self.c_attn(x)  # (B, T, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each (B, T, n_embd)

        head_size = self.n_embd // self.n_head
        # view: unpack the last dim into (n_head, head_size) while keeping time as T.
        # (B, T, n_embd) -> (B, T, n_head, head_size): still "time-major" (T next to head).
        # transpose(1, 2): swap T and n_head -> (B, n_head, T, head_size).
        # Why: torch.scaled_dot_product_attention expects (B, num_heads, seq_len, head_dim)
        # so each head is a contiguous (T, head_size) slab for batched GEMMs (QK^T, AV).
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        # q, k, v now (B, n_head, T, head_size)

        if self.cp_group is not None:
            # Ring attention in parallel/context.py is written for time-major head layout:
            # (B, T_local, n_head, head_dim). Undo the SDPA layout with one more transpose.
            if hasattr(self, '_ring_attn_fn'):
                ring_attention_forward = self._ring_attn_fn
            else:
                from parallel.context import ring_attention_forward
            q_cp = q.transpose(1, 2)  # (B, T, n_head, head_size)
            k_cp = k.transpose(1, 2)
            v_cp = v.transpose(1, 2)
            y = ring_attention_forward(q_cp, k_cp, v_cp, self.cp_group)  # (B, T, n_head, head_size)
            y = y.contiguous().view(B, T, self.n_embd)  # merge heads back into C
        else:
            # SDPA returns (B, n_head, T, head_size); transpose back to (B, T, n_head, head_size)
            # so we can merge head dims into C for the output Linear.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, n_head, T, head_size)
            y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)

        # c_proj expects (B, T, n_embd)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        aux_loss = 0.0
        for block in self.transformer.h:
            if hasattr(block, 'is_moe_block') and block.is_moe_block:
                x, block_aux_loss = block(x)
                aux_loss = aux_loss + block_aux_loss
            else:
                x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if isinstance(aux_loss, torch.Tensor):
                loss = loss + self.config.aux_loss_coeff * aux_loss
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# ---------------------------------------------------------------------------
# Parallelism application functions
# ---------------------------------------------------------------------------

def apply_tensor_parallelism(model, tp_group):
    """
    Replace linear layers in attention and MLP with tensor-parallel versions.

    For each transformer block:
      - Attention: c_attn (QKV projection) -> ColumnParallelLinear
                   c_proj (output projection) -> RowParallelLinear
      - MLP:       c_fc -> ColumnParallelLinear
                   c_proj -> RowParallelLinear

    ============================================================
    TODO [Part 2]: Implement this function
    ============================================================

    Steps:
      1. Import ColumnParallelLinear and RowParallelLinear from parallel.tensor
      2. For each block in model.transformer.h:
         a. Replace block.attn.c_attn with ColumnParallelLinear(n_embd, 3*n_embd, tp_group=tp_group)
         b. Replace block.attn.c_proj with RowParallelLinear(n_embd, n_embd, tp_group=tp_group)
         c. Update block.attn.n_head to n_head // tp_size (each rank handles fewer heads)
         d. Replace block.mlp.c_fc with ColumnParallelLinear(n_embd, 4*n_embd, tp_group=tp_group)
         e. Replace block.mlp.c_proj with RowParallelLinear(4*n_embd, n_embd, tp_group=tp_group)

    Important notes:
      - The fused QKV projection (c_attn) outputs 3*n_embd. With column-parallel,
        each rank gets 3*n_embd//tp_size outputs, which naturally gives each rank
        its share of Q, K, and V heads.
      - After replacing c_attn, the attention forward's split(self.n_embd, dim=2)
        must use n_embd // tp_size instead. Update block.attn.n_embd accordingly.
      - Copy the appropriate weight slices from the original layers to the new ones.
    """
    raise NotImplementedError("TODO [Part 2]: Implement apply_tensor_parallelism")


def apply_context_parallelism(model, cp_group):
    """
    Enable context parallelism (Ring Attention) in the model.

    This sets the cp_group on each attention module so that
    the forward pass uses ring_attention_forward instead of
    standard scaled_dot_product_attention.

    ============================================================
    TODO [Part 4]: Implement this function
    ============================================================

    Steps:
      1. For each block in model.transformer.h:
         a. Set block.attn.cp_group = cp_group
      2. That's it! The CausalSelfAttention.forward() already checks
         self.cp_group and dispatches to ring_attention_forward.
    """
    raise NotImplementedError("TODO [Part 4]: Implement apply_context_parallelism")


def apply_moe(model, config, ep_group=None):
    """
    Replace the MLP in selected transformer blocks with MoE layers.

    ============================================================
    TODO [Part 5]: Implement this function
    ============================================================

    Steps:
      1. Import MoELayer from parallel.expert
      2. For each block index i in range(config.n_layer):
         - If i % config.moe_freq == 1 (every other layer, starting from layer 1):
           a. Replace block.mlp with MoELayer(config, ep_group=ep_group)
           b. Set block.is_moe_block = True  (flag for forward pass)
           c. The block's forward must return (x, aux_loss) for MoE blocks
      3. Modify Block.forward for MoE blocks to handle the extra aux_loss output.

    Notes:
      - With n_layer=12 and moe_freq=2, layers 1,3,5,7,9,11 become MoE layers.
      - The MoELayer has the same input/output dimensions as the original MLP.
    """
    raise NotImplementedError("TODO [Part 5]: Implement apply_moe")
