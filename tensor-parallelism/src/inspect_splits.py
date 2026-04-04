import argparse
import logging
import os

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)

MODELS = {
    "gpt": "model_gpt_tp",
    "llama": "model_llama_tp",
}


def classify_param(
    name: str, module_path: str, param: torch.nn.Parameter, ws: int
) -> tuple[list[int], str, bool]:
    """Return (full_shape, note, is_split) for a parameter."""
    full = list(param.shape)
    is_weight = name.endswith("weight")

    if "ColumnParallelLinear" in module_path:
        if is_weight:
            full[0] *= ws
            return full, "col-parallel (d_out sharded)", True
        full[0] *= ws
        return full, "col-parallel bias", True

    if "RowParallelLinear" in module_path:
        if is_weight:
            full[1] *= ws
            return full, "row-parallel (d_in sharded)", True
        return full, "row-parallel bias (rank 0 only)", False

    return full, "replicated (not split)", False


def get_module_class_map(block: torch.nn.Module) -> dict[str, str]:
    """Map parameter prefix -> module class name."""
    result = {}
    for mod_name, mod in block.named_modules():
        cls = type(mod).__name__
        result[mod_name] = cls
    return result


def find_parent_class(param_name: str, class_map: dict[str, str]) -> str:
    """Find the class name of the module that owns a parameter."""
    parts = param_name.rsplit(".", 1)
    if len(parts) == 2:
        mod_prefix = parts[0]
        if mod_prefix in class_map:
            return class_map[mod_prefix]
    return ""


def inspect_block(model: torch.nn.Module, ws: int) -> None:
    block = model.blocks[0]
    class_map = get_module_class_map(block)

    sep = "=" * 114
    dash = "-" * 112

    logger.info("")
    logger.info(sep)
    logger.info("  WEIGHT SHAPES - Block 0 - TP=%d", ws)
    logger.info(sep)
    logger.info(
        "  %-7s %-35s %-18s %-15s %s", "Status", "Layer", "This GPU", "Full", "Note"
    )
    logger.info("  %s", dash)

    sharded_count = 0
    replicated_count = 0

    for name, p in block.named_parameters():
        parent_cls = find_parent_class(name, class_map)
        full, note, is_split = classify_param(name, parent_cls, p, ws)
        tag = "[SPLIT]" if is_split else "[FULL] "
        logger.info(
            "  %s %-35s %-18s %-15s %s",
            tag,
            name,
            str(list(p.shape)),
            str(full),
            note,
        )

        if is_split:
            sharded_count += p.numel()
        else:
            replicated_count += p.numel()

    total_gpu = sharded_count + replicated_count
    total_full = sharded_count * ws + replicated_count

    logger.info(sep)
    logger.info(
        "  On this GPU: %s | Full block: %s | Ratio: %.1f%%",
        f"{total_gpu:,}",
        f"{total_full:,}",
        total_gpu / total_full * 100,
    )
    logger.info(
        "  Sharded: %s (%.1f%%) | Replicated: %s (%.1f%%)",
        f"{sharded_count:,}",
        sharded_count / total_gpu * 100,
        f"{replicated_count:,}",
        replicated_count / total_gpu * 100,
    )
    logger.info(sep)
    logger.info("")


def parse_args():
    p = argparse.ArgumentParser(description="Inspect TP weight splits for block 0")
    p.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="gpt",
        help="which TP model to inspect (default: gpt)",
    )
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
    )

    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(device)

    rank = dist.get_rank()
    ws = dist.get_world_size()

    module_name = MODELS[args.model]
    mod = __import__(module_name)

    if args.model == "gpt":
        model = mod.TPGPT().to(device)
    else:
        model = mod.TPLlama().to(device)

    if rank == 0:
        inspect_block(model, ws)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
