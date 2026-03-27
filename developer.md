# Training Parallelism Strategies From Scratch

Hands-on implementations of distributed training parallelism strategies using native PyTorch,
deployed via SageMaker HyperPod PyTorchJob CRDs on Kubernetes.

## Project Structure

```
training-parallelism-strategies-from-scratch/
  tensor-parallelism/           Megatron-LM style TP (ColPar/RowPar, f/f* conjugates)
    src/                        Working implementations
      model.py                  Standard GPT baseline (d=512, 8 heads, 6 layers, ~29M params)
      tp.py                     TP primitives: ColumnParallelLinear, RowParallelLinear, TPGPT
      test_model.py             Single-GPU smoke test
    solutions/                  Full reference implementations (GQA, SP, TP+FSDP, inference)

  docker/                       Container images
    Dockerfile.cuda-12.9-pytorch-2.8-py3.12     Base: CUDA 12.9 + PyTorch 2.8 + EFA + Flash Attention + TE
    Dockerfile.cuda-12.8-pytorch-2.7-py3.12     Base: CUDA 12.8 + PyTorch 2.7 + EFA + Flash Attention + TE
    Dockerfile.experiments      Thin app layer on top of base (fast rebuild, source files only)

  k8s/                          Kubernetes manifests
    hpto_tp.j2.yaml             Jinja2 template for HyperPodPyTorchJob
    hpto_job.yaml               Reference manifest (production training job)

  scripts/                      Automation
    build_image.sh              Build + push the experiment image
    run_hpto_job.sh             Render template + submit HyperPodPyTorchJob
```

## Setup

```bash
uv sync
```

For development tools (black, isort, flake8):

```bash
uv sync --extra dev
```

## Docker Images

Two-layer image strategy: a heavy base image (rarely rebuilt) and a thin experiment
image (rebuilt in seconds when source files change).

### Base image (rebuild only when dependencies change)

```bash
cd docker/
docker build -f Dockerfile -t "mlp.docker.zooxlabs.com/asaha/dist-train/cuda-12.9-pytorch-2.8-py3.12:1.0.0-rc1" .
docker push "mlp.docker.zooxlabs.com/asaha/dist-train/cuda-12.9-pytorch-2.8-py3.12:1.0.0-rc1"
```

Base image includes: CUDA 12.9, PyTorch 2.8, Flash Attention 2.8.2, Transformer Engine 2.3,
EFA + GDRCopy for inter-node NCCL, HyperPod elastic agent.

libfabric	2.1.0amzn5.0 (libfabric1-aws)
aws-ofi-nccl	1.16.2 (libnccl-ofi_1.16.2-1)
nccl 2.27.3

### Experiment image (rebuild when source files change)

Using the build script (recommended -- uses content-addressed SHA tags):

```bash
./scripts/build_image.sh              # build + push
./scripts/build_image.sh --no-push    # build only
```

Or manually:

```bash
docker build -f docker/Dockerfile.experiments -t "mlp.docker.zooxlabs.com/asaha/dist-train/experiments:1.0.0-rc1" .
docker push "mlp.docker.zooxlabs.com/asaha/dist-train/experiments:1.0.0-rc1"
```

### Test locally

```bash
docker run --rm --gpus 1 mlp.docker.zooxlabs.com/asaha/dist-train/experiments:1.0.0-rc1
```

## Kubernetes Deployment

Jobs run on SageMaker HyperPod via `HyperPodPyTorchJob` CRDs. The `run_hpto_job.sh`
script handles building the image, rendering the Jinja2 template, and submitting to K8s.

### Quick start

```bash
# Smoke test: run test_model.py on 1 p5en node (8x H200)
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp-smoke \
    --node-type p5en \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py
    
# Smoke test: run test_model.py on 1 p4d node (8x A100)
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp-smoke \
    --node-type p4d \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py

# Skip rebuild, reuse last image
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp-smoke \
    --node-type p5en \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py

# Use a specific image
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp-smoke \
    --node-type p5en \
    --image-uri mlp.docker.zooxlabs.com/asaha/dist-train/experiments:1.0.0-rc1

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


# Run on 2 nodes. The yaml template guarantees that pods will be on differnt nodes.
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

## References

- [Megatron-LM paper](https://arxiv.org/abs/1909.08053) -- original TP formulation
- [Megatron-LM v3](https://arxiv.org/abs/2205.05198) -- sequence parallelism
- [GQA paper](https://arxiv.org/abs/2305.13245) -- grouped query attention
