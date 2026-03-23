from model import *

if __name__ == "__main__":
    model = StandardGPT().cuda()
    n_params = count_parameters(model)

    print(f'Model parameters: {n_params:,} ({n_params * 2 / 1e6:.1f} MB in BF16)')
    print(f'Architecture: d_model={D_MODEL}, n_heads={N_HEADS}, d_ff={D_FF}, layers={N_LAYERS}')
    print(f'\nBlock 0 parameter breakdown:')
    for name, param in model.named_parameters():
        if 'blocks.0.' in name:
            print(f'  {name:40s} {str(list(param.shape)):20s} {param.numel():>8,}')

    test_input = torch.randint(0, VOCAB_SIZE, (2, 32)).cuda()
    with torch.no_grad():
        out = model(test_input)
    print(f'\nForward pass OK: {list(test_input.shape)} -> {list(out.shape)}')
    del model; torch.cuda.empty_cache()
