#!/usr/bin/env python3
"""Download a HuggingFace model to a local cache directory.

Reads MODEL_ID from the environment (or --model-id CLI arg).
Cache location is controlled by HF_HOME / HUGGINGFACE_HUB_CACHE env vars.
"""

import argparse
import logging
import os
import time

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a HuggingFace model")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", ""),
        help="HuggingFace model ID (default: $MODEL_ID env var)",
    )
    args = parser.parse_args()

    model_id = args.model_id
    if not model_id:
        parser.error("--model-id is required (or set MODEL_ID env var)")

    cache_dir = os.environ.get("HF_HOME", os.environ.get("HUGGINGFACE_HUB_CACHE", ""))
    log.info("Downloading %s to %s", model_id, cache_dir or "(default cache)")

    t0 = time.time()
    path = snapshot_download(model_id)
    elapsed = time.time() - t0

    log.info("Done in %.1fs -> %s", elapsed, path)


if __name__ == "__main__":
    main()
