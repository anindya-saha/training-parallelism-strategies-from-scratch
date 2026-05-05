#!/usr/bin/env python3
"""
Naive 2-rank pipeline ("naive model parallelism").

Each **optimizer step** runs one full minibatch of size ``global_batch_size`` (``gbs``)
through the partition; there are no microbatches inside a step. This script uses
``num_samples=64`` and ``gbs=16``, so the loop runs **4** steps over the data.

Uses ``model_gpt.StandardGPT``: ``get_stage`` builds the full model on ``meta``,
takes ``list(full_model.layers)``, and gives each rank a contiguous slice (same idea
as ``gpipe.py`` / ``pipedream.py`` and ``model.get_stage`` on ``ToyModel.layers``).
``dist.broadcast`` shares ``input_ids`` with the last rank for ``lm_loss``.

``gpipe.py`` uses the same ``num_samples`` / ``gbs`` per outer step but splits each
``gbs`` into ``n_micro`` microbatches (all forwards, then all backwards per step).

Run:
  cd pipeline-parallelism/src && torchrun --standalone --nproc_per_node=2 naive.py
"""

import logging
import os

import torch
import torch.distributed as dist
from gpt2_model import (
    DEFAULT_D_FF,
    DEFAULT_D_MODEL,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    DEFAULT_VOCAB_SIZE,
    get_stage,
    lm_loss,
    pipeline_model_config_string,
)

logger = logging.getLogger(__name__)


def configure_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s [rank {rank}] %(name)s: %(message)s",
    )



def setup() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank = dist.get_rank()
    ws = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, ws, device


def cleanup() -> None:
    dist.destroy_process_group()


def main() -> None:
    rank, world_size, device = setup()

    configure_logging(rank)

    if world_size != 2:
        raise RuntimeError("This script expects exactly 2 processes.")

    torch.manual_seed(42)
    seq_len = 32
    d_model = DEFAULT_D_MODEL
    vocab_size = DEFAULT_VOCAB_SIZE

    num_samples = 64
    gbs = 16
    lr = 0.01

    if num_samples % gbs != 0:
        raise ValueError("num_samples must be divisible by global_batch_size")

    num_steps = num_samples // gbs

    stage, _, _ = get_stage(rank, world_size, device)

    optim = torch.optim.SGD(stage.parameters(), lr=lr)

    if rank == 0:
        all_input_ids = torch.randint(
            0,
            vocab_size,
            (num_samples, seq_len),
            device=device,
            dtype=torch.long,
        )
        logger.info(
            "Naive GPT: %s | num_samples=%d, gbs=%d, seq_len=%d, num_steps=%d",
            pipeline_model_config_string(
                d_model,
                DEFAULT_N_HEADS,
                DEFAULT_D_FF,
                DEFAULT_N_LAYERS,
                vocab_size,
                DEFAULT_MAX_SEQ_LEN,
            ),
            num_samples,
            gbs,
            seq_len,
            num_steps,
        )
    else:
        all_input_ids = torch.empty(
            num_samples, seq_len, device=device, dtype=torch.long
        )

    dist.broadcast(all_input_ids, src=0)

    for step in range(num_steps):
        start = step * gbs
        input_ids = all_input_ids[start : start + gbs].contiguous()

        optim.zero_grad()

        if rank == 0:
            h0 = stage(input_ids)
            dist.send(h0.detach().contiguous(), dst=1)
            grad_h = torch.empty(gbs, seq_len, d_model, device=device)
            dist.recv(grad_h, src=1)
            h0.backward(grad_h)
        else:
            buf = torch.empty(gbs, seq_len, d_model, device=device)
            dist.recv(buf, src=0)
            h1_req = buf.clone().requires_grad_(True)
            logits = stage(h1_req)
            loss = lm_loss(logits, input_ids)
            loss.backward()
            assert h1_req.grad is not None
            dist.send(h1_req.grad.detach().contiguous(), dst=0)

        optim.step()

    dist.barrier()
    if rank == 0:
        logger.info(
            "Done: %s steps, global_batch_size=%s, num_samples=%s.",
            num_steps,
            gbs,
            num_samples,
        )

    cleanup()


if __name__ == "__main__":
    main()
