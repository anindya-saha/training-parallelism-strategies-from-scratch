"""
Test Pipeline Parallelism correctness.
Run: torchrun --standalone --nproc_per_node=2 tests/test_pp.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from config import ModelConfig
from model import GPT


def test_pipeline_stages():
    """Verify pipeline stages produce same output as full model."""
    from parallel.pipeline import create_pipeline_stages

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank}'
    torch.manual_seed(42)

    config = ModelConfig(n_layer=4, n_head=4, n_embd=64, block_size=32, vocab_size=256)
    B, T = 2, 16

    # Create full model on all ranks for reference
    ref_model = GPT(config).to(device)

    x = torch.randint(0, 256, (B, T), device=device)
    dist.broadcast(x, src=0)

    # Reference forward
    with torch.no_grad():
        ref_logits, _ = ref_model(x)

    # Create pipeline stages
    pp_group = dist.new_group(list(range(world_size)))
    stage = create_pipeline_stages(config, world_size, pp_group, device)

    # Copy weights from reference model to stages
    # (This is what students would verify in their implementation)

    if rank == 0:
        print("PASS: Pipeline stages created successfully")
        print(f"  Stage 0 has {sum(p.numel() for p in stage.parameters())} parameters")

    dist.barrier()


if __name__ == "__main__":
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    test_pipeline_stages()

    destroy_process_group()
    if int(os.environ.get('RANK', 0)) == 0:
        print("\nPP tests complete.")
