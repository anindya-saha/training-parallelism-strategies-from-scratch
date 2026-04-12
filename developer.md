# Training Parallelism Strategies From Scratch

Hands-on implementations of distributed training parallelism strategies using native PyTorch,
deployed via SageMaker HyperPod PyTorchJob CRDs on Kubernetes.

## Project Structure

```
training-parallelism-strategies-from-scratch/
  dev.env                         Environment-specific config (gitignored)
  prod.env                         Environment-specific config (gitignored)

  tensor-parallelism/             Megatron-LM style TP (ColPar/RowPar, f/f* conjugates)
    src/                          Working implementations
      model_gpt.py                GPT baseline (no TP)
      model_gpt_tp.py             GPT with TP
      model_llama.py              Llama baseline (no TP)
      model_llama_tp.py           Llama with TP
      test_model.py               Unified smoke test (GPT + Llama)
      tp_primitives.py            TP primitive tests (Column/Row Linear)
      tp_inference_72b.py         72B inference (pure TP and TP+DP)
      tp_scaling_study.py         Throughput/batch scaling experiments
      benchmark_no_tp.py          Single-GPU benchmark harness
      compare_results.py          Result comparison + plots
      inspect_splits.py           NCCL split inspector
      measure_allreduce.py        AllReduce latency measurement
    solutions/                    Full reference implementations (GQA, SP, TP+FSDP, inference)

  docker/                         Container images
    Dockerfile.experiments        App layer on top of AWS DLC base image

  k8s/                            Kubernetes manifests
    hpto_job.j2.yaml              Jinja2 template for HyperPodPyTorchJob
    model_download.j2.yaml        Jinja2 template for model download K8s Job
    jupyter_lab.deployment.j2.yaml  Jupyter Lab Deployment + Service (Jinja2)
    hpto_fsdp.yaml                Reference FSDP manifest

  scripts/                        Automation
    build_image.sh                Build + push the experiment image
    run_hpto_job.sh               Render template + submit HyperPodPyTorchJob
    run_model_download.sh         Render template + submit model download Job
    run_jupyter_lab.sh            Render Jupyter YAML, delete if exists, kubectl create
```

## Setup

```bash
uv sync
```

For development tools (black, isort, flake8):

```bash
uv sync --extra dev
```

## Environment configuration

Environment-specific variables (registry, namespace, HF token, cache paths)
live in your `dev.env`. Source it before running any script:

```bash
source dev.env
```

The scripts (`run_hpto_job.sh`, `run_model_download.sh`, `run_jupyter_lab.sh`) expect variables like
`REGISTRY`, `NAMESPACE`, `SERVICE_ACCOUNT`, `FSX_CLAIM`, `HF_CACHE_DIR`, and
`HF_TOKEN` to already be in the environment.

## Creating Docker Images

The experiment image is a thin layer on top of the
[AWS Deep Learning Container](https://github.com/aws/deep-learning-containers)
for PyTorch. 

It adds `hyperpod-elastic-agent`, `transformers`, `rich`,
`jupyterlab`, `ipykernel`, `matplotlib`, and `deepspeed`. 

It copies
`tensor-parallelism/src` (Python files), `pipeline-parallelism/` (notebooks and
code), and `scripts/download_model.py` into `/workspace/training-parallelism-strategies-from-scratch`

It exposes port 8888 for Jupyter Lab when you run that workload from Kubernetes
(see `k8s/jupyter_lab.deployment.j2.yaml` and `scripts/run_jupyter_lab.sh`).

Base image: `public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2`

Using the build script (recommended - uses content-addressed SHA tags):

```bash
./scripts/build_image.sh              # build + push
./scripts/build_image.sh --no-push    # build only
```

Or manually:

```bash
docker build -f docker/Dockerfile.experiments -t ${REGISTRY}/${USER}/dist-train/experiments:latest .
docker push ${REGISTRY}/${USER}/dist-train/experiments:latest
```

Test locally:

```bash
docker run --rm --gpus 1 ${REGISTRY}/${USER}/dist-train/experiments:latest
```

Make sure `REGISTRY` is set in your environment (via `dev.env`) before running.

## Kubernetes Deployment

Jobs run on SageMaker HyperPod via `HyperPodPyTorchJob` CRDs. The `run_hpto_job.sh`
script renders the Jinja2 template and submits to K8s. Environment variables
(`NAMESPACE`, `SERVICE_ACCOUNT`, `HF_TOKEN`, etc.) must be in the environment.
Source your `dev.env` first.

### Model download

See [tp_inference_72b.md](tp_inference_72b.md#prerequisites-download-the-model-to-fsx)
for model download commands (`run_model_download.sh`, manual, and Argo Workflows).

### Quick start

```bash
# Smoke test: run test_model.py on 1 p5en node (8x H200). Skip rebuild, reuse last image.
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp-smoke \
    --node-type p5en \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py

# Smoke test: run test_model.py on 1 p4d node (8x A100). Skip rebuild, reuse last image.
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp-smoke \
    --node-type p4d \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py

# Use a specific image
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp-smoke \
    --node-type p5en \
    --image-uri ${REGISTRY}/${IMAGE_NAME}:${IMAGE_VERSION}

# Smoke test: build, push, and run test_model.py on 1 p5en node (8x H200)
./scripts/run_hpto_job.sh \
    --job-name tp-smoke \
    --node-type p5en \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py

# Preview rendered YAML without submitting
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp-smoke \
    --node-type p5en \
    --dry-run

# Run on 2 nodes (anti-affinity forces pods onto separate nodes)
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp-nvlink-test \
    --node-type p5en \
    --num-nodes 2 \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py
```

### Node type presets

| Preset | Instance Type         | GPUs           |
|--------|-----------------------|----------------|
| p4d    | ml.p4d.24xlarge       | 8x A100 40GB   |
| p4de   | ml.p4de.24xlarge      | 8x A100 80GB   |
| p5en   | ml.p5en.48xlarge      | 8x H200 141GB  |
| p6     | ml.p6-b200.48xlarge   | 8x B200        |

### Jupyter Lab

The experiment image includes Jupyter Lab. `scripts/run_jupyter_lab.sh` renders
`k8s/jupyter_lab.deployment.j2.yaml` to `<deployment-name>.yaml` in the current
working directory (same pattern as `run_hpto_job.sh` writing `<job-name>.yaml`),
runs `kubectl delete -f` on that file if it exists (drops prior Deployment + Service),
then `kubectl create -f`.
You connect from your laptop with `kubectl port-forward` to the Service on port 8888.

**Prerequisites:** `source dev.env`, working `kubectl` context to the cluster, and
`jinja2` CLI (`uv sync` in this repo).

**1. Deploy Jupyter** (`--node-type` is required; same presets as `run_hpto_job.sh`,
or a full label such as `ml.p5en.48xlarge`). By default `run_jupyter_lab.sh` runs
`build_image.sh` (build + push), matching `run_hpto_job.sh`. Use `--skip-build` to
reuse the newest local image without rebuilding, or `--image-uri` for a specific
tag (implies `--skip-build`).

```bash
./scripts/run_jupyter_lab.sh --node-type p5en

./scripts/run_jupyter_lab.sh --node-type p5en --skip-build

./scripts/run_jupyter_lab.sh \
    --node-type p5en \
    --image-uri "${REGISTRY}/${USER}/dist-train/experiments:<tag>"
```

For build without push, then deploy with that tag:
`eval "$(./scripts/build_image.sh --no-push)"` then
`./scripts/run_jupyter_lab.sh --node-type p5en --image-uri "${IMAGE_URI}"`.

If you omit `--jupyter-token`, the script generates one and prints it to stderr:
save that value for the browser. Optional flags: `--deployment-name` (default
`jupyter-lab`), `--gpu-count` (default `4`), `--namespace`, `--dry-run` to print
rendered YAML only.

**2. Wait until the pod is ready:**

```bash
kubectl get pods -n "${NAMESPACE:-mlp}" -l "app=jupyter-lab,deployment=jupyter-lab"
kubectl wait pod -n "${NAMESPACE:-mlp}" -l "app=jupyter-lab,deployment=jupyter-lab" --for=condition=Ready --timeout=300s
```

Use your deployment name in the label if you passed `--deployment-name`.

**3. Port-forward from your machine** (default Service name is `<deployment>-svc`):

```bash
kubectl port-forward -n "${NAMESPACE:-mlp}" "svc/jupyter-lab-svc" 8888:8888
```

**4. Open Jupyter Lab** in a browser, using the token from step 1:

`http://127.0.0.1:8888/lab?token=<YOUR_TOKEN>`

**Cleanup** (default deployment name `jupyter-lab`; change names if you used
`--deployment-name`):

```bash
kubectl delete -n "${NAMESPACE:-mlp}" "deployment/jupyter-lab" "svc/jupyter-lab-svc"
```

### Monitoring a running job

```bash
kubectl get hyperpodpytorchjob <job-name> -n mlp
kubectl get pods -n mlp -l job-name=<job-name>
kubectl logs -f -l job-name=<job-name> -n mlp --all-containers
kubectl describe hyperpodpytorchjob <job-name> -n mlp
```

### Cleanup

```bash
kubectl delete hyperpodpytorchjob <job-name> -n mlp
```

## Benchmarks

### Single GPU (no parallelism)

```bash
# GPT baseline
python tensor-parallelism/src/model_gpt.py

# Llama baseline
python tensor-parallelism/src/model_llama.py

# Custom model size (GPT example)
python tensor-parallelism/src/model_gpt.py \
    --d-model 1024 --n-heads 16 --d-ff 4096 --n-layers 12

# Custom model size (Llama example - n_kv_heads for GQA)
python tensor-parallelism/src/model_llama.py \
    --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 12

# Custom benchmark settings
python tensor-parallelism/src/model_gpt.py \
    --batch-size 16 --seq-len 512 --warmup 5 --benchmark 20
```

Results are written to `results_model_gpt.json` / `results_model_llama.json`.

### Tensor parallelism

```bash
# GPT TP on 4 GPUs
torchrun --nproc_per_node=4 tensor-parallelism/src/model_gpt_tp.py

# Llama TP on 4 GPUs
torchrun --nproc_per_node=4 tensor-parallelism/src/model_llama_tp.py

# 8 GPUs with larger model (GPT)
torchrun --nproc_per_node=8 tensor-parallelism/src/model_gpt_tp.py \
    --d-model 1024 --n-heads 16 --d-ff 4096 --n-layers 12

# 8 GPUs with larger model (Llama - n_kv_heads must be divisible by nproc)
torchrun --nproc_per_node=8 tensor-parallelism/src/model_llama_tp.py \
    --d-model 1024 --n-heads 16 --n-kv-heads 8 --n-layers 12
```

Results are written to `results_model_gpt_tp.json` / `results_model_llama_tp.json`.

### TP primitives test

```bash
torchrun --nproc_per_node=2 tensor-parallelism/src/tp_primitives.py
```

### On Kubernetes (via HyperPodPyTorchJob)

```bash
# GPT single-GPU baseline on p5en
./scripts/run_hpto_job.sh --skip-build \
    --job-name bench-gpt-baseline \
    --node-type p5en \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/src/model_gpt.py

# GPT TP benchmark on p5en (8x H200)
./scripts/run_hpto_job.sh --skip-build \
    --job-name bench-gpt-tp \
    --node-type p5en \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/src/model_gpt_tp.py

# Llama single-GPU baseline on p5en
./scripts/run_hpto_job.sh --skip-build \
    --job-name bench-llama-baseline \
    --node-type p5en \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/src/model_llama.py

# Llama TP benchmark on p5en (8x H200)
./scripts/run_hpto_job.sh --skip-build \
    --job-name bench-llama-tp \
    --node-type p5en \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/src/model_llama_tp.py
```

### 72B inference: pure TP and TP+DP

See [tp_inference_72b.md](tp_inference_72b.md) for local benchmarks, Kubernetes
deployment commands, topology diagrams, and NCCL log analysis.

## Architecture

### TP Communication Pattern

```
TP splits individual weight matrices across GPUs:

  Column-Parallel (expanding layers):     Row-Parallel (contracting layers):
    W_q, W_k, W_v  -- split heads          W_o         -- split input dim
    W1 (FFN up)     -- split d_ff           W2 (FFN down) -- split input dim
    No communication in forward             ALL-REDUCE in forward

  Communication per transformer block: 2 all-reduces
    1. After attention (W_o row-parallel)
    2. After FFN (W2 row-parallel)

  Replicated (NOT split): LayerNorm, embeddings
```

### Sequence Parallelism (SP)

```
Without SP:                         With SP:
  LN input: (B, T, d) replicated     LN input: (B, T/N, d) seq-split
  -> all-reduce after W_o             -> all-gather before Q/K/V
  -> all-reduce after W2              -> reduce-scatter after W_o/W2

  Same total comm volume, but LN/residual activations use 1/N memory.
```

### GQA (Grouped Query Attention)

```
MHA: Q(8 heads), K(8 heads), V(8 heads)   -- all heads independent
GQA: Q(16 heads), K(4 heads), V(4 heads)  -- 4 Q heads share each KV head

With TP=4 on LARGE_CONFIG (16 Q, 4 KV heads):
  GPU 0: Q heads 0-3,  KV head 0  (repeat KV 4x before attention)
  GPU 1: Q heads 4-7,  KV head 1
  GPU 2: Q heads 8-11, KV head 2
  GPU 3: Q heads 12-15, KV head 3
```

### 2D Parallelism (TP + FSDP)

```
2 nodes x 4 GPUs/node = 8 GPUs total:

  mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

  Node 0: [GPU 0 -- GPU 1 -- GPU 2 -- GPU 3]   TP group (NVLink)
  Node 1: [GPU 4 -- GPU 5 -- GPU 6 -- GPU 7]   TP group (NVLink)
            |         |         |         |
            +-------- FSDP (cross-node) ---+

  TP handles compute splitting within a node.
  FSDP handles memory optimization across nodes.
```

## Key Results (from Vizuara tutorial on 2x H200)

```
                         No TP (1 GPU)     TP (2 GPUs)
  Parameters/GPU            29,405,184      17,401,856  (40.8% fewer)
  Model memory (MB)              113.1            66.8  (40.9% less)
  Peak memory (MB)              1204.4           837.4  (30.5% less)
  Full step (ms)                 11.84           19.19
  Efficiency                       ---           30.8%
```

TP reduces memory but adds communication overhead. Efficiency improves with
larger models where compute dominates over all-reduce latency.

## References

- [Megatron-LM paper](https://arxiv.org/abs/1909.08053) - original TP formulation
- [Megatron-LM v3](https://arxiv.org/abs/2205.05198) - sequence parallelism
- [GQA paper](https://arxiv.org/abs/2305.13245) - grouped query attention
- Vizuara GPU Engineering Course - source material for from-scratch implementations
