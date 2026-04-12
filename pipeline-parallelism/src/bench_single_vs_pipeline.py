#!/usr/bin/env python3
"""Compare one-GPU ``StandardGPT`` vs a 2-rank naive pipeline (same init, same batch).

Rank 0 builds one randomly initialized ``StandardGPT``, copies its ``state_dict`` to CPU,
broadcasts it to all ranks, and each rank loads its pipeline slice (same layout as
``naive.py``). Then:

1. **Forward:** max absolute difference between full-model logits and pipeline logits.
2. **One SGD step:** max absolute difference between full-model weights after one step
   vs merged pipeline stage weights after one manual pipeline step (same as ``naive.py``).

Run (2 GPUs, NCCL):

  cd pipeline-parallelism/src && torchrun --standalone --nproc_per_node=2 bench_single_vs_pipeline.py

Optional: ``--atol 1e-5`` style overrides for pass/fail (defaults are loose for FP32).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from gpt2_model import (
    DEFAULT_D_MODEL,
    DEFAULT_N_LAYERS,
    DEFAULT_VOCAB_SIZE,
    StandardGPT,
    get_stage,
    lm_loss,
    load_stage_from_full_state_dict,
    merge_pipeline_stage_state_dicts_to_full,
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


def _max_abs_state_dict_diff(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]
) -> tuple[float, str | None]:
    worst = 0.0
    worst_key: str | None = None
    for k in a:
        if k not in b:
            return float("inf"), k
        d = (a[k] - b[k]).abs().max().item()
        if d > worst:
            worst = d
            worst_key = k
    return worst, worst_key


def pipeline_naive_one_step(
    rank: int,
    stage: nn.Sequential,
    optim: torch.optim.SGD,
    input_ids: torch.Tensor,
    d_model: int,
) -> None:
    gbs, seq_len = input_ids.shape
    device = input_ids.device
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Micro batch size for the comparison (all ranks use the same tensor).",
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--forward-atol",
        type=float,
        default=5e-5,
        help="Fail if max |logits_full - logits_pipe| exceeds this.",
    )
    parser.add_argument(
        "--weights-atol",
        type=float,
        default=1e-4,
        help="Fail if max |state_dict_ref - state_dict_merged| exceeds this.",
    )
    args = parser.parse_args()

    rank, world_size, device = setup()
    configure_logging(rank)

    if world_size != 2:
        logger.error("Need exactly 2 processes (torchrun --nproc_per_node=2).")
        cleanup()
        sys.exit(1)

    torch.manual_seed(args.seed)

    # --- Shared full weights on CPU (rank 0 builds, then broadcast) ---
    full_sd_cpu: dict[str, torch.Tensor] | None = None
    if rank == 0:
        ref = StandardGPT().to(device)
        full_sd_cpu = {k: v.detach().cpu().clone() for k, v in ref.state_dict().items()}
        del ref
        torch.cuda.empty_cache()

    payload: list[dict[str, torch.Tensor] | None] = [full_sd_cpu]
    dist.broadcast_object_list(payload, src=0)
    assert payload[0] is not None
    full_sd_cpu = payload[0]

    stage, g_start, _g_end = get_stage(
        rank,
        world_size,
        device,
        quiet=True,
        log_full=False,
        log_stage=False,
    )
    load_stage_from_full_state_dict(
        stage, full_sd_cpu, g_start, device=device
    )

    gbs = args.batch_size
    seq_len = args.seq_len
    vocab_size = DEFAULT_VOCAB_SIZE
    d_model = DEFAULT_D_MODEL

    if rank == 0:
        input_ids = torch.randint(
            0, vocab_size, (gbs, seq_len), device=device, dtype=torch.long
        )
    else:
        input_ids = torch.empty(gbs, seq_len, device=device, dtype=torch.long)
    dist.broadcast(input_ids, src=0)

    stage.eval()
    dist.barrier()

    # --- Forward: pipeline vs full (full only on rank 0) ---
    logits_pipe = torch.empty(gbs, seq_len, vocab_size, device=device)
    if rank == 0:
        h0 = stage(input_ids)
        dist.send(h0.detach().contiguous(), dst=1)
        dist.recv(logits_pipe, src=1)
    else:
        buf = torch.empty(gbs, seq_len, d_model, device=device)
        dist.recv(buf, src=0)
        logits_pipe_local = stage(buf)
        dist.send(logits_pipe_local.detach().contiguous(), dst=0)

    forward_max: float | None = None
    loss_diff: float | None = None
    if rank == 0:
        full_eval = StandardGPT().to(device)
        full_eval.load_state_dict(
            {k: v.to(device) for k, v in full_sd_cpu.items()}, strict=True
        )
        full_eval.eval()
        with torch.no_grad():
            logits_full = full_eval(input_ids)
        forward_max = (logits_full - logits_pipe).abs().max().item()
        loss_full = lm_loss(logits_full, input_ids)
        loss_pipe = lm_loss(logits_pipe, input_ids)
        loss_diff = (loss_full - loss_pipe).abs().item()
        logger.info(
            "Forward: max abs logits diff = %.3e | loss diff = %.3e",
            forward_max,
            loss_diff,
        )

    stage.train()
    dist.barrier()

    # --- One SGD step: single GPU on rank 0 vs pipeline on both ranks ---
    if rank == 0:
        full_train = StandardGPT().to(device)
        full_train.load_state_dict(
            {k: v.to(device) for k, v in full_sd_cpu.items()}, strict=True
        )
        opt_ref = torch.optim.SGD(full_train.parameters(), lr=args.lr)
        opt_ref.zero_grad()
        loss_ref = lm_loss(full_train(input_ids), input_ids)
        loss_ref.backward()
        opt_ref.step()
        sd_after_ref = {
            k: v.detach().cpu() for k, v in full_train.state_dict().items()
        }

    # Reload pipeline stages from same initial weights (fresh graph)
    load_stage_from_full_state_dict(
        stage, full_sd_cpu, g_start, device=device
    )
    optim_pipe = torch.optim.SGD(stage.parameters(), lr=args.lr)
    pipeline_naive_one_step(rank, stage, optim_pipe, input_ids, d_model)

    sd0_cpu: dict[str, torch.Tensor] | None = None
    if rank == 0:
        sd0_cpu = {k: v.detach().cpu() for k, v in stage.state_dict().items()}

    sd1_payload: list[dict[str, torch.Tensor] | None] = [None]
    if rank == 1:
        sd1_payload[0] = {k: v.detach().cpu() for k, v in stage.state_dict().items()}
    dist.broadcast_object_list(sd1_payload, src=1)
    sd1_cpu = sd1_payload[0]
    assert sd1_cpu is not None

    weight_max: float | None = None
    weight_worst_key: str | None = None
    if rank == 0:
        assert sd0_cpu is not None
        n_mod = 1 + DEFAULT_N_LAYERS + 2
        layer_chunk = (n_mod + world_size - 1) // world_size
        s1_start = layer_chunk
        merged = merge_pipeline_stage_state_dicts_to_full(
            [(sd0_cpu, g_start), (sd1_cpu, s1_start)]
        )
        weight_max, weight_worst_key = _max_abs_state_dict_diff(
            sd_after_ref, merged
        )
        logger.info(
            "After 1x SGD: max abs weight diff = %.3e (worst key %r)",
            weight_max,
            weight_worst_key,
        )

    # Broadcast pass/fail scalar from rank 0 for clean exit code
    ok_tensor = torch.zeros(1, device=device, dtype=torch.int32)
    if rank == 0:
        assert forward_max is not None and loss_diff is not None
        assert weight_max is not None
        ok = (
            forward_max <= args.forward_atol
            and loss_diff <= args.forward_atol
            and weight_max <= args.weights_atol
        )
        ok_tensor[0] = 1 if ok else 0
    dist.broadcast(ok_tensor, src=0)
    all_ok = int(ok_tensor.item()) == 1

    if rank == 0:
        if all_ok:
            logger.info(
                "PASS: forward atol <= %.3e and weight atol <= %.3e",
                args.forward_atol,
                args.weights_atol,
            )
        else:
            logger.error(
                "FAIL: forward max=%.3e (tol %.3e), loss diff=%.3e, weight max=%.3e (tol %.3e)",
                forward_max,
                args.forward_atol,
                loss_diff,
                weight_max,
                args.weights_atol,
            )

    cleanup()
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
