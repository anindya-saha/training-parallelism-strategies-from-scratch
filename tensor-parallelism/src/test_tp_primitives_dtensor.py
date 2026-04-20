"""TP primitive checks using torch.distributed.tensor.parallel (DTensor).

Aligned with tp_primitives.py and the same diagram:

  X  = [[0,1],[2,3],[4,5],[6,7]]   (4x2)
  W1 = [[1,3],[2,4]]               (2x2)   used as X @ W1 in the diagram
  W2 = [[5,7],[6,8]]               (2x2)   used as H @ W2 with H = X @ W1
  Y  = X @ W1 @ W2 = [[34,46],[148,200],[262,354],[376,508]]  (4x2)

nn.Linear computes y = x @ weight.T + bias, so we set weight = W1.T and W2.T.

Run: torchrun --standalone --nproc_per_node=2 test_tp_primitives_dtensor.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.distributed.tensor import Shard, Replicate


# Diagram matrices (same as tp_primitives.TPPrimitivesTest)
def reference_X() -> torch.Tensor:
    return torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.float32)


def reference_W1() -> torch.Tensor:
    return torch.tensor([[1, 3], [2, 4]], dtype=torch.float32)


def reference_W2() -> torch.Tensor:
    return torch.tensor([[5, 7], [6, 8]], dtype=torch.float32)


def reference_Y() -> torch.Tensor:
    return reference_X() @ reference_W1() @ reference_W2()


class ColLinearOnly(nn.Module):
    """Single column-parallel linear: y = x @ W1 (diagram convention)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(2, 2, bias=False)
        W1 = reference_W1()
        with torch.no_grad():
            self.lin1.weight.copy_(W1.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin1(x)


class RowLinearOnly(nn.Module):
    """Single row-parallel linear: y = x @ W2 (diagram convention)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin2 = nn.Linear(2, 2, bias=False)
        W2 = reference_W2()
        with torch.no_grad():
            self.lin2.weight.copy_(W2.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(x)


class ColThenRow(nn.Module):
    """Column then row: y = (x @ W1) @ W2 (no gather/scatter between)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(2, 2, bias=False)
        self.lin2 = nn.Linear(2, 2, bias=False)
        W1 = reference_W1()
        W2 = reference_W2()
        with torch.no_grad():
            self.lin1.weight.copy_(W1.T)
            self.lin2.weight.copy_(W2.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.lin1(x))


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


def test_column_linear_only(
    mesh: DeviceMesh, device: torch.device, rank: int, ws: int
) -> None:
    print(
        f"\n{'=' * 70}\n  DTENSOR TEST 1: ColwiseParallel (single Linear)\n{'=' * 70}"
    )

    X = reference_X().to(device)
    W1 = reference_W1().to(device)

    m = ColLinearOnly().to(device)
    # Same as tp_primitives._CopyToParallelRegion: each rank holds the full tensor.
    m = parallelize_module(
        m, mesh, {"lin1": ColwiseParallel(input_layouts=Replicate())}
    )
    Y_local = m(X)

    Y_local_expected = (X @ W1).chunk(ws, dim=-1)[rank]
    assert torch.allclose(Y_local, Y_local_expected, atol=1e-4), (
        Y_local,
        Y_local_expected,
    )

    gathered = [torch.zeros_like(Y_local) for _ in range(ws)]
    dist.all_gather(gathered, Y_local.contiguous())
    Y_full = torch.cat(gathered, dim=-1)

    Y_full_expected = X @ W1
    assert torch.allclose(Y_full, Y_full_expected, atol=1e-4), (Y_full, Y_full_expected)


def test_row_linear_only(
    mesh: DeviceMesh, device: torch.device, rank: int, ws: int
) -> None:
    print(
        f"\n{'=' * 70}\n  DTENSOR TEST 2: RowwiseParallel (single Linear)\n{'=' * 70}"
    )

    X = reference_X().to(device)
    W2 = reference_W2().to(device)

    # Same as tp_primitives._ScatterToParallelRegion: each rank holds one last-dim chunk.
    X_local = X.chunk(ws, dim=-1)[rank].contiguous()

    m = RowLinearOnly().to(device)
    # Default input_layouts=Shard(-1): local tensor is already this rank's shard.
    m = parallelize_module(m, mesh, {"lin2": RowwiseParallel(input_layouts=Shard(-1))})

    Y_full = m(
        X_local
    )  # same as tp_primitives._ReduceFromParallelRegion: sum partials -> full Y

    Y_full_expected = X @ W2
    assert torch.allclose(Y_full, Y_full_expected, atol=1e-4), (Y_full, Y_full_expected)


def test_column_then_row(
    mesh: DeviceMesh, device: torch.device, rank: int, ws: int
) -> None:
    print(
        f"\n{'=' * 70}\n  DTENSOR TEST 3: ColwiseParallel -> RowwiseParallel\n{'=' * 70}"
    )

    X = reference_X().to(device)
    Y_full_expected = reference_Y().to(device)

    m = ColThenRow().to(device)
    parallelize_module(
        m,
        mesh,
        {"lin1": ColwiseParallel(), "lin2": RowwiseParallel()},
    )
    Y_full = m(X)
    assert torch.allclose(Y_full, Y_full_expected, atol=1e-4), (Y_full, Y_full_expected)


def main() -> None:
    device, rank, ws = setup_distributed()
    try:
        if ws != 2:
            if rank == 0:
                print(f"ERROR: need exactly 2 GPUs for this test, got ws={ws}")
            return

        mesh = init_device_mesh("cuda", (ws,), mesh_dim_names=("tp",))

        if rank == 0:
            print("=" * 70)
            print("  TP primitives (DTensor) - same reference as test_tp_primitives.py")
            X = reference_X()
            W1 = reference_W1()
            W2 = reference_W2()
            Y = reference_Y()
            print(
                f"  X{list(X.shape)} @ W1{list(W1.shape)} @ W2{list(W2.shape)} = Y{list(Y.shape)}"
            )
            print("=" * 70)

        dist.barrier()
        test_column_linear_only(mesh, device, rank, ws)
        dist.barrier()
        test_row_linear_only(mesh, device, rank, ws)
        dist.barrier()
        test_column_then_row(mesh, device, rank, ws)
        dist.barrier()

        if rank == 0:
            print(f"\n{'=' * 70}\n  ALL DTENSOR TP PRIMITIVE TESTS PASSED\n{'=' * 70}")
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
