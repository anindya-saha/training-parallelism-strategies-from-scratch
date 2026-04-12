#!/usr/bin/env python3
"""
PipeDream-Flush / 1F1B (one-forward-one-backward) 2-rank pipeline schedule.

Uses ``get_stage`` from ``model_gpt.py`` with the same partition as ``gpipe.py`` and
``naive.py`` (each rank gets a contiguous slice of ``full_model.layers``). ``dist.broadcast``
replicates ``all_input_ids`` for ``lm_loss`` on the last stage.

This script uses ``num_samples=64``, ``gbs=16``, ``n_micro=4``, ``mbs=4``, and **4**
outer steps, matching the batching constants in those scripts.

Schedule (for 2 stages, m microbatches):
  - Warmup:  rank 0 does (n-1)=1 forward without a matching backward.
  - Steady:  each iteration does 1 backward then 1 forward (rank 0), or
             recv → forward → backward → send grad (rank 1).
  - Cooldown: rank 0 drains the remaining 1 backward.

Compared to GPipe (all forwards, then all backwards), 1F1B limits the number of
in-flight activations to ``num_warmup + 1`` on the first stage, rather than ``m``.
For 2 stages and 4 microbatches, peak stored activations on rank 0 is 2 (vs 4 in GPipe).

Run:
  cd pipeline-parallelism/src && torchrun --standalone --nproc_per_node=2 pipedream.py

Or:
  cd pipeline-parallelism/src && python -m torch.distributed.run --standalone --nproc_per_node=2 pipedream.py
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
    n_stages = world_size  # 2
    num_warmup = n_stages - 1 - rank  # rank 0: 1, rank 1: 0
    num_steady = n_micro - num_warmup

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
            "PipeDream-Flush GPT: num_samples=%d, gbs=%d, n_micro=%d, mbs=%d, "
            "num_steps=%d, num_warmup=%d, num_steady=%d, seq_len=%d, max_seq_len=%d",
            num_samples,
            gbs,
            n_micro,
            mbs,
            num_steps,
            num_warmup,
            num_steady,
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
        fwd_count = 0
        bwd_count = 0

        optim.zero_grad()

        if rank == 0:
            for _ in range(num_warmup):
                h0 = stage(micro_inputs[fwd_count])
                saved_h0.append(h0)
                dist.send(h0.detach().contiguous(), dst=1)
                fwd_count += 1

            for _ in range(num_steady):
                grad_h = torch.empty(mbs, seq_len, d_model, device=device)
                dist.recv(grad_h, src=1)
                saved_h0[bwd_count].backward(grad_h)
                bwd_count += 1

                h0 = stage(micro_inputs[fwd_count])
                saved_h0.append(h0)
                dist.send(h0.detach().contiguous(), dst=1)
                fwd_count += 1

            for _ in range(num_warmup):
                grad_h = torch.empty(mbs, seq_len, d_model, device=device)
                dist.recv(grad_h, src=1)
                saved_h0[bwd_count].backward(grad_h)
                bwd_count += 1

            optim.step()

        else:
            for micro_b in range(n_micro):
                buf = torch.empty(mbs, seq_len, d_model, device=device)
                dist.recv(buf, src=0)
                h1_req = buf.clone().requires_grad_(True)
                logits = stage(h1_req)
                loss_b = lm_loss(logits, micro_inputs[micro_b])
                loss_b.backward()
                assert h1_req.grad is not None
                dist.send(h1_req.grad.detach().contiguous(), dst=0)

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
