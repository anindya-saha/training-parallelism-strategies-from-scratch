"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.

Usage:
  python prepare.py
  python prepare.py --data_root /path/to/edu_fineweb10B
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger("prepare")

REMOTE_NAME = "sample-10BT"
SHARD_SIZE = int(1e8)  # 100M tokens per shard

_enc: tiktoken.Encoding | None = None
_eot: int | None = None


def _pool_init_worker() -> None:
    """Each worker loads its own tokenizer (safe for fork and spawn)."""
    global _enc, _eot
    _enc = tiktoken.get_encoding("gpt2")
    _eot = _enc._special_tokens["<|endoftext|>"]


def tokenize(doc: dict) -> np.ndarray:
    """Tokenize a single document; returns uint16 token ids."""
    assert _enc is not None and _eot is not None
    tokens = [_eot]
    tokens.extend(_enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename: str, tokens_np: np.ndarray) -> None:
    np.save(filename, tokens_np)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and tokenize FineWeb-Edu to shards"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="edu_fineweb10B",
        help="Directory to write token shards (created if missing). Relative paths are resolved against cwd.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    data_cache_dir = os.path.abspath(args.data_root)
    os.makedirs(os.path.join(data_cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_cache_dir, "val"), exist_ok=True)
    logger.info("data_root=%s", data_cache_dir)

    logger.info(
        "loading dataset HuggingFaceFW/fineweb-edu name=%s split=train", REMOTE_NAME
    )
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=REMOTE_NAME, split="train")

    nprocs = max(1, os.cpu_count() // 2)
    logger.info("using %d worker processes for tokenization", nprocs)

    with mp.Pool(nprocs, initializer=_pool_init_worker) as pool:
        shard_index = 0
        all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
        token_count = 0
        progress_bar: tqdm | None = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < SHARD_SIZE:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=SHARD_SIZE,
                        unit="tokens",
                        desc=f"Shard {shard_index}",
                    )
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    data_cache_dir, split, f"edufineweb_{shard_index:06d}.npy"
                )
                remainder = SHARD_SIZE - token_count
                assert progress_bar is not None
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                write_datafile(filename, all_tokens_np)
                logger.info("wrote shard %s (%d tokens)", filename, SHARD_SIZE)
                shard_index += 1
                progress_bar.close()
                progress_bar = None
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if progress_bar is not None:
            progress_bar.close()

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                data_cache_dir, split, f"edufineweb_{shard_index:06d}.npy"
            )
            write_datafile(filename, all_tokens_np[:token_count])
            logger.info("wrote final shard %s (%d tokens)", filename, token_count)

    logger.info("done")


if __name__ == "__main__":
    main()
