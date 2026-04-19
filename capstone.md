## Environment configuration

Environment-specific variables (registry, namespace, HF token, cache paths)
live in your `env.dev`. Source it before running any script:

```bash
source env.dev
```

## Creating Docker Images (capstone DLC image)

The capstone training image is defined under **`capstone/containers/`**: it builds a
**DLC-style** PyTorch image (CUDA on Amazon Linux 2023, EFA, optional extras from
`docker/pytorch/pyproject.toml`, including **`--extra capstone`** and **`--extra capstone-hyperpod`**
in the `runtime-base` stage). It does **not** bake your `capstone/src` tree into the image;
mount code at run time (for example FSX or a ConfigMap) or add a small `COPY` layer on top if you want.

**This is not** `scripts/build_image.sh` in the repo root. That script builds **`docker/Dockerfile.experiments`**
(the thin **tensor-parallelism** experiment layer on a public AWS DLC base). Use it only for jobs that
expect that layout. For capstone, build from **`capstone/containers`**:

```bash
cd ~/training-parallelism-strategies-from-scratch/capstone/containers

DOCKER_BUILDKIT=1 docker build \
  -f docker/pytorch/Dockerfile \
  --target runtime \
  -t "${REGISTRY}/${USER}/dist-train:capstone" \
  .
```

Then push the tag you passed to `-t` (for example `docker push ...`) after `source env.dev` (or your `dev.env`) so **`REGISTRY`** is set.

### Python optional groups (`docker/pytorch/pyproject.toml`)

These extras are resolved in the **same** `uv.lock` (run `uv lock` from `capstone/containers/docker/pytorch/` after editing deps):

- **`capstone`** — workshop runtime (datasets, tiktoken, transformers, …)
- **`capstone-hyperpod`** — `hyperpod-elastic-agent`
- **`capstone-notebook`** — Jupyter stack
- **`capstone-dev`** — black, isort, flake8

The Docker **`runtime-base`** stage installs **`capstone`** and **`capstone-hyperpod`** only; add more `--extra` flags in the Dockerfile if you need notebook or dev inside the image.

### Quick start

From the **repo root** (so `./scripts/run_hpto_job.sh` resolves). Ensure your job image or volume mounts lay out `capstone/src` under the paths below.

```
rsync -av ~/training-parallelism-strategies-from-scratch/capstone/src/ /mnt/fsx-static/asaha/training-parallelism-strategies-from-scratch/capstone/src/ \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__'
```

```bash
# Smoke test: run train.py on 1 p5en node (8x H200). Skip rebuild, reuse last image.
./scripts/run_hpto_job.sh --skip-build \
    --job-name dist-train \
    --node-type p5en \
    --train-script /mnt/fsx/asaha/training-parallelism-strategies-from-scratch/capstone/src/train.py \
    --train-args "--data_root /mnt/fsx/asaha/parallelism-experiments/data/edu_fineweb10B --max_steps 100"

# Smoke test: run train.py on 1 p4d node (8x A100). Skip rebuild, reuse last image.
./scripts/run_hpto_job.sh --skip-build \
    --job-name dist-train \
    --node-type p4d \
    --train-script /mnt/fsx/asaha/training-parallelism-strategies-from-scratch/capstone/src/train.py \
    --train-args "--data_root /mnt/fsx/asaha/parallelism-experiments/data/edu_fineweb10B --max_steps 10"
