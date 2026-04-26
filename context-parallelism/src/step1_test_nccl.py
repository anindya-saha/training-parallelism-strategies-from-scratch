"""Step 1: Verify NCCL works on fresh pod"""

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    ws = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    logger.info("[GPU %s] Initialized (device=%s)", rank, device)

    x = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(x)

    expected = ws * (ws + 1) / 2
    logger.info(
        "[GPU %s] all_reduce result: %s (expected: %s)",
        rank,
        x.item(),
        expected,
    )

    dist.barrier()

    if rank == 0:
        logger.info("✓NCCL is working. Proceed to step 2.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
