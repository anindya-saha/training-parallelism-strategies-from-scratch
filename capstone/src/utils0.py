"""
Utility functions for 5D parallelism process group creation and logging.
"""
import torch
import torch.distributed as dist
from config import ParallelConfig


def create_process_groups(parallel_config: ParallelConfig):
    """
    Create process groups for 5D parallelism.

    Given a world of GPUs, organize them into a multi-dimensional grid
    and create communicator groups for each parallelism dimension.

    The rank layout follows the convention (outermost to innermost):
        DP > PP > CP > EP > TP
    TP is innermost because it requires the highest bandwidth (NVLink).

    ============================================================
    TODO [Part 6]: Implement this function
    ============================================================

    Example with TP=2, PP=2, DP=2 on 8 GPUs:
        Rank grid (DP x PP x TP):
            DP=0: [[0, 1], [2, 3]]   (PP=0: ranks 0,1; PP=1: ranks 2,3)
            DP=1: [[4, 5], [6, 7]]   (PP=0: ranks 4,5; PP=1: ranks 6,7)

        TP groups:  {0,1}, {2,3}, {4,5}, {6,7}
        PP groups:  {0,2}, {1,3}, {4,6}, {5,7}
        DP groups:  {0,4}, {1,5}, {2,6}, {3,7}

    Steps:
      1. Get world_size from dist and compute the 5D grid shape:
         (dp_size, pp_size, cp_size, ep_size, tp_size)
      2. Create a rank tensor: torch.arange(world_size).reshape(dp, pp, cp, ep, tp)
      3. For each dimension, iterate over all other dimensions' indices and
         extract the rank list along that dimension -> dist.new_group(ranks)
      4. Return the group that this rank belongs to for each dimension.

    Args:
        parallel_config: ParallelConfig with tp_size, pp_size, cp_size, ep_size, dp_size

    Returns:
        dict with keys 'tp', 'pp', 'cp', 'ep', 'dp', each mapping to a process group.
        If a dimension has size 1, its group is None.
    """
    raise NotImplementedError("TODO [Part 6]: Implement create_process_groups")


def get_data_parallel_rank(groups):
    """Get this process's rank within the data parallel group."""
    if groups.get('dp') is None:
        return 0
    return dist.get_rank(groups['dp'])


def get_data_parallel_world_size(groups):
    """Get the number of data parallel replicas."""
    if groups.get('dp') is None:
        return 1
    return dist.get_world_size(groups['dp'])
