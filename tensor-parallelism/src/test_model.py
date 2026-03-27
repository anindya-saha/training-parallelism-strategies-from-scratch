"""Smoke test for GPT and Llama model architectures."""

import torch

from model_gpt import (
    D_FF, D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE,
    StandardGPT, count_parameters,
)
from model_llama import (
    D_FF_LLAMA, N_KV_HEADS,
    StandardLlama,
)


def test_model(name, model, arch_info):
    n_params = count_parameters(model)
    print(f'\n{"=" * 60}')
    print(f'{name}')
    print(f'{"=" * 60}')
    print(f'Parameters: {n_params:,} ({n_params * 2 / 1e6:.1f} MB in BF16)')
    print(f'Architecture: {arch_info}')
    print(f'\nBlock 0 parameter breakdown:')
    for pname, param in model.named_parameters():
        if 'blocks.0.' in pname:
            print(f'  {pname:40s} {str(list(param.shape)):20s} {param.numel():>8,}')

    test_input = torch.randint(0, VOCAB_SIZE, (2, 32)).cuda()
    with torch.no_grad():
        out = model(test_input)
    print(f'\nForward pass OK: {list(test_input.shape)} -> {list(out.shape)}')
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_model(
        "GPT (Standard MHA + LayerNorm + GELU FFN)",
        StandardGPT().cuda(),
        f'd_model={D_MODEL}, n_heads={N_HEADS}, d_ff={D_FF}, layers={N_LAYERS}',
    )
    test_model(
        "Llama (GQA + RMSNorm + RoPE + SwiGLU FFN)",
        StandardLlama().cuda(),
        f'd_model={D_MODEL}, n_heads={N_HEADS}, n_kv_heads={N_KV_HEADS}, '
        f'd_ff={D_FF_LLAMA}, layers={N_LAYERS}',
    )
