#!/usr/bin/env python3
"""
GPipe-style 2-rank pipeline (manual cross-rank backward).

Uses ``model_gpt.StandardGPT`` and ``get_stage``: each rank holds a contiguous slice
of ``list(full_model.layers)`` on ``device``. Activations cross ranks with ``detach`` /
``send``; gradients are exchanged explicitly for ``backward`` (no distributed autograd).
``dist.broadcast`` replicates ``all_input_ids`` so both ranks slice the same microbatches
for ``lm_loss`` on the last stage.

Schedule: for each optimizer step, all microbatch forwards, then all microbatch
backwards. Here ``num_samples=64``, ``gbs=16``, ``n_micro=4``, ``mbs=4``, so each step
covers 16 samples in one ``gbs`` and the outer loop runs **4** steps (same totals as
``naive.py`` for one pass over the 64 samples).

For the naive one-forward-one-backward-per-minibatch schedule, see ``naive.py``.

Run:
  cd pipeline-parallelism/src && torchrun --standalone --nproc_per_node=2 gpipe.py

Or:
  cd pipeline-parallelism/src && python -m torch.distributed.run --standalone --nproc_per_node=2 gpipe.py
"""

import logging
import os

import torch
import torch.distributed as dist
from gpt2_model import (
    DEFAULT_D_MODEL,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_VOCAB_SIZE,
    get_stage,
    lm_loss,
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

    n_micro = 4
    mbs = gbs // n_micro

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
            "GPipe GPT: num_samples=%d, gbs=%d, n_micro=%d, mbs=%d, num_steps=%d, "
            "seq_len=%d, max_seq_len=%d",
            num_samples,
            gbs,
            n_micro,
            mbs,
            num_steps,
            seq_len,
            DEFAULT_MAX_SEQ_LEN,
        )
    else:
        all_input_ids = torch.empty(
            num_samples, seq_len, device=device, dtype=torch.long
        )

    dist.broadcast(all_input_ids, src=0)

    for step in range(num_steps):
        start = step * gbs
        micro_inputs = [
            all_input_ids[start + b * mbs : start + (b + 1) * mbs].contiguous()
            for b in range(n_micro)
        ]

        saved_h0: list[torch.Tensor] = []
        acts1: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        if rank == 0:
            for b in range(n_micro):
                h0 = stage(micro_inputs[b])
                saved_h0.append(h0)
                dist.send(h0.detach().contiguous(), dst=1)
        else:
            for b in range(n_micro):
                buf = torch.empty(mbs, seq_len, d_model, device=device)
                dist.recv(buf, src=0)
                h1_req = buf.clone().requires_grad_(True)
                logits = stage(h1_req)
                acts1.append((h1_req, logits, micro_inputs[b]))

        if rank == 1:
            optim.zero_grad()
            for b in range(n_micro - 1, -1, -1):
                h1_req, logits, ids_b = acts1[b]
                loss_b = lm_loss(logits, ids_b)
                loss_b.backward()
                assert h1_req.grad is not None
                dist.send(h1_req.grad.detach().contiguous(), dst=0)
            optim.step()

        if rank == 0:
            optim.zero_grad()
            for b in range(n_micro - 1, -1, -1):
                grad_h = torch.empty(mbs, seq_len, d_model, device=device)
                dist.recv(grad_h, src=1)
                saved_h0[b].backward(grad_h)
            optim.step()

    dist.barrier()
    if rank == 0:
        logger.info(
            "Done: %d steps, gbs=%d, n_micro=%d, mbs=%d.",
            num_steps,
            gbs,
            n_micro,
            mbs,
        )
    cleanup()


if __name__ == "__main__":
    main()
