"""Trace activation shapes through Vanilla TP vs TP+SP."""

import os
import torch.distributed as dist

from config import *


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank = dist.get_rank()
    ws = dist.get_world_size()

    B, S, h = BATCH_SIZE, SEQ_LEN, D_MODEL
    S_local = S // ws
    h_local = h // ws
    dff_local = D_FF // ws
    heads_local = N_HEADS // ws

    if rank == 0:
        print()
        print("=" * 90)
        print(f"  ACTIVATION SHAPES: Vanilla TP vs TP + SP  (N={ws} GPUs)")
        print("=" * 90)
        print(f"  Config: B={B}, S={S}, h={h}, d_ff={D_FF}, n_heads={N_HEADS}")
        print(
            f"  Per-GPU: S/N={S_local}, h/N={h_local}, d_ff/N={dff_local}, heads/N={heads_local}"
        )
        print("=" * 90)
        print()

        header = (
            f"  {'Location':<50} {'Vanilla TP':<22} {'TP + SP':<22} {'Savings':<10}"
        )
        print(header)
        print("  " + "-" * 100)

        # Define shapes at each location
        shapes = [
            # Location, Vanilla TP shape, TP+SP shape, Savings
            (
                "Block input",
                f"(B,S,h)={B}x{S}x{h}",
                f"(B,S/N,h)={B}x{S_local}x{h}",
                f"1/{ws}",
            ),
            (
                "LayerNorm 1 input",
                f"(B,S,h)={B}x{S}x{h}",
                f"(B,S/N,h)={B}x{S_local}x{h}",
                f"1/{ws}",
            ),
            (
                "LayerNorm 1 output",
                f"(B,S,h)={B}x{S}x{h}",
                f"(B,S/N,h)={B}x{S_local}x{h}",
                f"1/{ws}",
            ),
            ("", "", "", ""),
            (">>> ALL-GATHER (SP to TP)", "---", f"(B,S,h)={B}x{S}x{h}", "temp"),
            ("", "", "", ""),
            (
                "Q, K, V (TP region)",
                f"(B,S,h/N)={B}x{S}x{h_local}",
                f"(B,S,h/N)={B}x{S}x{h_local}",
                "same",
            ),
            (
                "Attention scores",
                f"(B,H/N,S,S)={B}x{heads_local}x{S}x{S}",
                f"same",
                "same",
            ),
            (
                "Attention output",
                f"(B,S,h/N)={B}x{S}x{h_local}",
                f"(B,S,h/N)={B}x{S}x{h_local}",
                "same",
            ),
            (
                "W_o output (partial)",
                f"(B,S,h)={B}x{S}x{h}",
                f"(B,S,h)={B}x{S}x{h}",
                "partial",
            ),
            ("", "", "", ""),
            (">>> ALL-REDUCE (Vanilla)", f"(B,S,h)={B}x{S}x{h}", "---", "---"),
            (
                ">>> REDUCE-SCATTER (TP+SP)",
                "---",
                f"(B,S/N,h)={B}x{S_local}x{h}",
                "---",
            ),
            ("", "", "", ""),
            (
                "Dropout",
                f"(B,S,h)={B}x{S}x{h}",
                f"(B,S/N,h)={B}x{S_local}x{h}",
                f"1/{ws}",
            ),
            (
                "Residual add",
                f"(B,S,h)={B}x{S}x{h}",
                f"(B,S/N,h)={B}x{S_local}x{h}",
                f"1/{ws}",
            ),
            ("", "", "", ""),
            (
                "LayerNorm 2 output",
                f"(B,S,h)={B}x{S}x{h}",
                f"(B,S/N,h)={B}x{S_local}x{h}",
                f"1/{ws}",
            ),
            ("", "", "", ""),
            (">>> ALL-GATHER (SP to TP)", "---", f"(B,S,h)={B}x{S}x{h}", "temp"),
            ("", "", "", ""),
            (
                "W1 output (TP region)",
                f"(B,S,d_ff/N)={B}x{S}x{dff_local}",
                f"same",
                "same",
            ),
            ("GeLU output", f"(B,S,d_ff/N)={B}x{S}x{dff_local}", f"same", "same"),
            (
                "W2 output (partial)",
                f"(B,S,h)={B}x{S}x{h}",
                f"(B,S,h)={B}x{S}x{h}",
                "partial",
            ),
            ("", "", "", ""),
            (">>> ALL-REDUCE (Vanilla)", f"(B,S,h)={B}x{S}x{h}", "---", "---"),
            (
                ">>> REDUCE-SCATTER (TP+SP)",
                "---",
                f"(B,S/N,h)={B}x{S_local}x{h}",
                "---",
            ),
            ("", "", "", ""),
            (
                "Dropout",
                f"(B,S,h)={B}x{S}x{h}",
                f"(B,S/N,h)={B}x{S_local}x{h}",
                f"1/{ws}",
            ),
            (
                "Block output",
                f"(B,S,h)={B}x{S}x{h}",
                f"(B,S/N,h)={B}x{S_local}x{h}",
                f"1/{ws}",
            ),
        ]

        for loc, van, sp, sav in shapes:
            if loc == "":
                print()
            else:
                print(f"  {loc:<50} {van:<22} {sp:<22} {sav:<10}")

        # Memory calculation
        print()
        print("=" * 90)
        print("  MEMORY SUMMARY (per layer, per GPU)")
        print("=" * 90)

        # Elements in SP region (saved by SP)
        sp_region_vanilla = (
            8 * B * S * h
        )  # LN1 in/out, dropout, residual, LN2 in/out, dropout, residual
        sp_region_tpsp = 8 * B * S_local * h

        # Elements in TP region (same for both)
        tp_region = (
            3 * B * S * h_local  # Q, K, V
            + B * heads_local * S * S  # attention scores
            + B * S * h_local  # attn output
            + 2 * B * S * dff_local  # W1, GeLU
        )

        vanilla_total = sp_region_vanilla + tp_region
        tpsp_total = sp_region_tpsp + tp_region

        print()
        print(f"  SP region (LayerNorm, Dropout, Residuals):")
        print(
            f"    Vanilla TP: {sp_region_vanilla:>12,} elements  ({sp_region_vanilla * 2 / 1024 / 1024:.2f} MB)"
        )
        print(
            f"    TP + SP:    {sp_region_tpsp:>12,} elements  ({sp_region_tpsp * 2 / 1024 / 1024:.2f} MB)"
        )
        print(f"    Savings:    {(1 - sp_region_tpsp/sp_region_vanilla)*100:.1f}%")
        print()
        print(f"  TP region (Q,K,V, attention, FFN):")
        print(
            f"    Both:       {tp_region:>12,} elements  ({tp_region * 2 / 1024 / 1024:.2f} MB)"
        )
        print()
        print(f"  TOTAL per layer:")
        print(
            f"    Vanilla TP: {vanilla_total:>12,} elements  ({vanilla_total * 2 / 1024 / 1024:.2f} MB)"
        )
        print(
            f"    TP + SP:    {tpsp_total:>12,} elements  ({tpsp_total * 2 / 1024 / 1024:.2f} MB)"
        )
        print(f"    Savings:    {(1 - tpsp_total/vanilla_total)*100:.1f}%")
        print()
        print(f"  Over {N_LAYERS} layers:")
        print(f"    Vanilla TP: {vanilla_total * N_LAYERS * 2 / 1024 / 1024:.1f} MB")
        print(f"    TP + SP:    {tpsp_total * N_LAYERS * 2 / 1024 / 1024:.1f} MB")
        print(
            f"    Savings:    {(vanilla_total - tpsp_total) * N_LAYERS * 2 / 1024 / 1024:.1f} MB"
        )
        print("=" * 90)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
