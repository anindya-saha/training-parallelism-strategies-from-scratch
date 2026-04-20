"""Verify TP primitives against hand-computed examples.

Uses the exact matrices from the hand-drawn diagram:

  X  = [[0,1],[2,3],[4,5],[6,7]]   (4x2)
  W1  = [[1,3],[2,4]]              (2x2)   used as X @ W1 in the diagram
  W2  = [[5,7],[6,8]]              (2x2)   used as H @ W2 with H = X @ W1
  Y  = X @ W1 @ W2 = [[34,46],[148,200],[262,354],[376,508]]  (4x2)

Run: torchrun --nproc_per_node=2 test_tp_primitives.py
"""

import os

import torch
import torch.distributed as dist


# Diagram matrices (same as tp_primitives.TPPrimitivesTest)
def reference_X() -> torch.Tensor:
    return torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.float32)


def reference_W1() -> torch.Tensor:
    return torch.tensor([[1, 3], [2, 4]], dtype=torch.float32)


def reference_W2() -> torch.Tensor:
    return torch.tensor([[5, 7], [6, 8]], dtype=torch.float32)


def reference_Y() -> torch.Tensor:
    return reference_X() @ reference_W1() @ reference_W2()


# ============================
#  TP Communication Primitives
# ============================


class _CopyToParallelRegion(torch.autograd.Function):
    """Identity in forward, all-reduce in backward.
    Placed BEFORE column-parallel layers.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce in forward, identity in backward.
    Placed AFTER row-parallel layers to combine partial sums.
    """

    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


class _ScatterToParallelRegion(torch.autograd.Function):
    """Chunk/scatter in forward, all-gather in backward.
    Placed BEFORE row-parallel layers to split the input across ranks.
    Conjugate of _AllGatherFromParallelRegion.
    """

    @staticmethod
    def forward(ctx, x):
        ws = dist.get_world_size()
        rank = dist.get_rank()
        ctx.ws = ws
        chunks = x.chunk(ws, dim=-1)
        return chunks[rank].contiguous()

    @staticmethod
    def backward(ctx, grad):
        gathered = [torch.zeros_like(grad) for _ in range(ctx.ws)]
        dist.all_gather(gathered, grad.contiguous())
        return torch.cat(gathered, dim=-1)


class _AllGatherFromParallelRegion(torch.autograd.Function):
    """All-gather in forward, chunk/scatter in backward.
    Placed AFTER column-parallel layers to reconstruct full output.
    Conjugate of _ScatterToParallelRegion.
    """

    @staticmethod
    def forward(ctx, y):
        ws = dist.get_world_size()
        ctx.ws = ws
        gathered = [torch.zeros_like(y) for _ in range(ws)]
        dist.all_gather(gathered, y.contiguous())
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        rank = dist.get_rank()
        chunks = grad.chunk(ctx.ws, dim=-1)
        return chunks[rank].contiguous()


def setup_distributed() -> tuple[torch.device, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    ws = dist.get_world_size()
    return device, rank, ws


def teardown_distributed() -> None:
    dist.destroy_process_group()


# ============================
#  Primitive Pairing Summary
# ============================
#
#  Standalone Column Linear:
#    _CopyToParallelRegion -> ColLinear -> _AllGatherFromParallelRegion
#    (identity fwd)                       (all-gather fwd)
#
#  Standalone Row Linear:
#    _ScatterToParallelRegion -> RowLinear -> _ReduceFromParallelRegion
#    (scatter fwd)                           (all-reduce fwd)
#
#  Combined Column + Row (what transformers actually use):
#    _CopyToParallelRegion -> ColLinear -> RowLinear -> _ReduceFromParallelRegion
#    (identity fwd)                                    (all-reduce fwd)
#
#    The all-gather and scatter CANCEL OUT between ColLinear and RowLinear
#    because ColLinear's split output is exactly what RowLinear expects
#    as its split input. This saves one communication round.


# class TPPrimitivesTest:
#     """Test harness for TP communication primitives.

#     Holds the diagram's reference matrices (X, W, Y) and provides
#     setup/teardown for the distributed process group.
#     """

#     def __init__(self):
#         self.X = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.float32)
#         self.W1 = torch.tensor([[1, 3], [2, 4]], dtype=torch.float32)
#         self.W2 = torch.tensor([[5, 7], [6, 8]], dtype=torch.float32)
#         self.Y = (self.X @ self.W1) @ self.W2
#         self.rank = None
#         self.ws = None

#     def setup(self):
#         dist.init_process_group("nccl")
#         self.rank = dist.get_rank()
#         self.ws = dist.get_world_size()
#         torch.cuda.set_device(self.rank)
#         self.X = self.X.cuda()
#         self.W1 = self.W1.cuda()
#         self.W2 = self.W2.cuda()
#         self.Y = self.Y.cuda()

#     def teardown(self):
#         dist.destroy_process_group()

# ------------------------------------------------------------------
# Test 1: Column-Parallel Linear
# ------------------------------------------------------------------
#   X (same on all GPUs) @ W_local (column shard) -> Y_local (split)
#   _AllGatherFromParallelRegion reconstructs full Y
# ------------------------------------------------------------------


def test_column_linear_only(device: torch.device, rank: int, ws: int):
    print(f"\n{'=' * 70}\n  TEST 1: Column-Parallel Linear\n{'=' * 70}")

    X = reference_X().to(device)
    W1 = reference_W1().to(device)

    W1_local = W1.chunk(ws, dim=1)[rank]

    X_local = _CopyToParallelRegion.apply(X)
    Y_local = X_local @ W1_local

    Y_local_expected = (X @ W1).chunk(ws, dim=1)[rank]
    assert torch.allclose(Y_local, Y_local_expected, atol=1e-4)

    Y_full = _AllGatherFromParallelRegion.apply(Y_local)

    Y_full_expected = X @ W1
    assert torch.allclose(Y_full, Y_full_expected, atol=1e-4), (Y_full, Y_full_expected)


# ------------------------------------------------------------------
# Test 2: Row-Parallel Linear
# ------------------------------------------------------------------
#   _ScatterToParallelRegion splits X across ranks
#   X_local @ W_local -> Y_partial
#   _ReduceFromParallelRegion sums partials -> full Y
# ------------------------------------------------------------------


def test_row_linear_only(device: torch.device, rank: int, ws: int):
    print(f"\n{'=' * 70}\n  TEST 2: Row-Parallel Linear\n{'=' * 70}")

    X = reference_X().to(device)
    W2 = reference_W2().to(device)

    X_local = _ScatterToParallelRegion.apply(X)
    W2_local = W2.chunk(ws, dim=0)[rank]

    Y_partial = X_local @ W2_local

    Y_full = _ReduceFromParallelRegion.apply(Y_partial)

    Y_full_expected = X @ W2
    assert torch.allclose(Y_full, Y_full_expected, atol=1e-4), (Y_full, Y_full_expected)


# ------------------------------------------------------------------
# Test 3: Column Linear -> Row Linear (combined)
# ------------------------------------------------------------------
#   The key insight: when Column and Row are chained, the
#   _AllGatherFromParallelRegion (after Column) and the
#   _ScatterToParallelRegion (before Row) cancel out.
#
#   Column output is already split -> Row expects split input.
#   No communication needed between them.
#
#   Only _CopyToParallelRegion at entry and
#   _ReduceFromParallelRegion at exit remain.
# ------------------------------------------------------------------


def test_column_then_row(device: torch.device, rank: int, ws: int):
    print(f"\n{'=' * 70}\n  TEST 3: Column Linear -> Row Linear (combined)\n{'=' * 70}")

    X = reference_X().to(device)
    W1 = reference_W1().to(device)
    W2 = reference_W2().to(device)

    Y_full_expected = reference_Y().to(device)

    # Entry: _CopyToParallelRegion (identity fwd)
    X_local = _CopyToParallelRegion.apply(X)

    # Column-parallel: split W1 by output columns
    W1_local = W1.chunk(ws, dim=1)[rank]
    Y_col_local = X_local @ W1_local

    Y_col_expected = (X @ W1).chunk(ws, dim=1)[rank]
    assert torch.allclose(Y_col_local, Y_col_expected, atol=1e-4)

    # NO all-gather or scatter here - they cancel out.
    # Y_col_local is already split, which is what Row Linear needs.

    # Row-parallel: split W2 by input rows
    W2_local = W2.chunk(ws, dim=0)[rank]
    Y_row_local = Y_col_local @ W2_local

    # Exit: _ReduceFromParallelRegion (all-reduce fwd)
    Y_full = _ReduceFromParallelRegion.apply(Y_row_local)

    assert torch.allclose(Y_full, Y_full_expected, atol=1e-4), (Y_full, Y_full_expected)

    # ------------------------------------------------------------------


def main() -> None:
    device, rank, ws = setup_distributed()
    try:
        if ws != 2:
            if rank == 0:
                print(f"ERROR: need exactly 2 GPUs for this test, got ws={ws}")
            return

        if rank == 0:
            print("=" * 70)
            print("  TP Primitives Test - matching hand-drawn diagram")
            X = reference_X()
            W1 = reference_W1()
            W2 = reference_W2()
            Y = reference_Y()
            print(
                f"  X{list(X.shape)} @ W1{list(W1.shape)} @ W2{list(W2.shape)} = Y{list(Y.shape)}"
            )
            print("=" * 70)

        dist.barrier()

        test_column_linear_only(device, rank, ws)
        dist.barrier()
        test_row_linear_only(device, rank, ws)
        dist.barrier()
        test_column_then_row(device, rank, ws)
        dist.barrier()

        if rank == 0:
            print(f"\n{'=' * 70}\n  ALL TP PRIMITIVE TESTS PASSED\n{'=' * 70}")
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
