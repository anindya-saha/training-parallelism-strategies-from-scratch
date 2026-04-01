# 72B Inference: Pure TP and TP+DP

End-to-end guide for running 72B model inference with Tensor Parallelism (TP)
and combined TP+DP (Data Parallelism) topologies, both locally and on
Kubernetes via AWS Sagemaker HyperPodPyTorchJob.

## Local benchmarks

```bash
# Pure TP=4 (1 replica, 4 GPUs)
# -> Runs TEST 1 (batch=100 OOM) + TEST 2 (TTFT/throughput)
torchrun --nproc_per_node=4 tensor-parallelism/src/tp_inference_72b.py --tp-size 4

# Pure TP=2 (1 replica, 2 GPUs - for comparison)
torchrun --nproc_per_node=2 tensor-parallelism/src/tp_inference_72b.py --tp-size 2

# Pure TP=8 (1 replica, 8 GPUs)
torchrun --nproc_per_node=8 tensor-parallelism/src/tp_inference_72b.py --tp-size 8

# TP=4 x DP=2 (2 replicas, 8 GPUs total)
# -> Runs TEST 3 (multi-replica serving throughput)
torchrun --nproc_per_node=8 tensor-parallelism/src/tp_inference_72b.py --tp-size 4

# TP=4 x DP=2 with custom prompts per replica
torchrun --nproc_per_node=8 tensor-parallelism/src/tp_inference_72b.py \
    --tp-size 4 \
    --prompts-per-replica 16

# Custom model (e.g. Llama 3.1 8B for quick testing)
torchrun --nproc_per_node=4 tensor-parallelism/src/tp_inference_72b.py \
    --tp-size 4 \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --max-new-tokens 64
```

Results are written to `results_tp{N}.json` or `results_tp{N}_dp{M}.json`.

## Kubernetes deployment (via HyperPodPyTorchJob)

Build and push the experiment image first (source `dev.env` for `REGISTRY`):

```bash
source dev.env
./scripts/build_image.sh            # build + push (content-addressed SHA tag)
./scripts/build_image.sh --no-push  # build only
```

All commands use `tp_inference_72b.py` with `--tp-size` to control the TP degree. When `world_size > tp_size`, the script automatically creates DP replicas. The `--gpus-per-node` flag controls both the K8s GPU resource request 
and `nproc_per_node` (1 GPU = 1 process).

| Key flags | GPUs | Topology |
|-----------|------|----------|
| `--gpus-per-node 4 --train-args "--tp-size 4 ..."` | 4 on 1 node | Pure TP=4, TEST 1+2 |
| `--train-args "--tp-size 8 ..."` | 8 on 1 node | Pure TP=8, TEST 1+2 |
| `--num-nodes 2 --gpus-per-node 4 --train-args "--tp-size 4 ..."` | 4+4 on 2 nodes | TP=4 x DP=2, TEST 3 |

All K8s commands write to a timestamped `--output-dir` on FSx
(e.g. `.../results/2026-03-22_143015/tp4/`) so results persist after
the pod terminates and runs never overwrite each other.

### Prerequisites: download the model to FSx

Before running inference, pre-download the model to shared storage so every pod
finds it cached. Use the standalone model download job.

```bash
# One-time: download Qwen2.5-72B to FSx
# (uses the latest local experiment image; polls until complete or 30 min timeout)
scripts/run_model_download.sh --model-id Qwen/Qwen2.5-72B-Instruct
```

Alternatively, download manually from any node that has FSx mounted.

Once cached, all inference jobs below will pick up the model from
`/mnt/fsx/asaha/hf_cache` without re-downloading.

### Running inference jobs

```bash
RESULTS_DIR="/mnt/fsx/asaha/parallelism-experiments/results/$(date +%Y-%m-%d_%H%M%S)"

# Pure TP=4 on a single p5en node (4 GPUs)
# -> 1 replica, 4 GPUs doing TP
# -> Runs TEST 1 (batch=100 OOM) + TEST 2 (TTFT/throughput)
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp4-inference \
    --node-type p5en \
    --gpus-per-node 4 \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/tp_inference_72b.py \
    --train-args "--tp-size 4 --output-dir ${RESULTS_DIR}/tp4"

# Pure TP=8 on a single p5en node (all 8 GPUs)
# -> Runs TEST 1 + TEST 2
./scripts/run_hpto_job.sh --skip-build \
    --job-name tp8-inference \
    --node-type p5en \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/tp_inference_72b.py \
    --train-args "--tp-size 8 --output-dir ${RESULTS_DIR}/tp8"

# TP=4 x DP=2 on 2 p5en nodes (4 GPUs per node, 8 total)
# -> 2 replicas, each with TP=4
# -> Each replica's 4 TP GPUs colocated on one node (NVLink)
# -> DP replicas on different nodes (EFA)
# -> Runs TEST 3 (multi-replica serving throughput)

./scripts/run_hpto_job.sh --skip-build \
    --job-name tp4-dp2-inference \
    --node-type p6 \
    --num-nodes 2 \
    --gpus-per-node 4 \
    --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/tp_inference_72b.py \
    --train-args "--tp-size 4 --prompts-per-replica 8 --output-dir ${RESULTS_DIR}/tp4-dp2"
```

### Topology for TP=4 x DP=2

```
  Node 0 (p5en)              Node 1 (p5en)
  +-----------------------+  +-----------------------+
  | GPU0 GPU1 GPU2 GPU3   |  | GPU0 GPU1 GPU2 GPU3   |
  |   Replica 0 (TP=4)    |  |   Replica 1 (TP=4)    |
  |   NVLink within node  |  |   NVLink within node   |
  +-----------+-----------+  +-----------+-----------+
              |                          |
              +------ EFA fabric --------+
              (no cross-node TP traffic; replicas are independent)
```

The TP=4 x DP=2 topology ensures each replica's 4 TP GPUs are colocated on
the same node (NVLink), while the two replicas sit on different nodes. The
`podAntiAffinity` rule in the template already forces pods onto separate nodes,
and each pod runs 4 processes (`--gpus-per-node 4`), so the 2D mesh
`(dp=2, tp=4)` maps replica 0 to node 0 and replica 1 to node 1. No cross-node
TP traffic - the only cross-node communication would be if you later added
gradient sync between replicas.

## NCCL Log Analysis: TP=4 x DP=2 on p6-b200.48xlarge

Verified from production logs of the `tp4-dp2-inference` job running on two
p6-b200.48xlarge nodes (Blackwell B200 GPUs), 4 GPUs per node, 8 GPUs total.

```
NAME                                               PF  READY   STATUS         RESTARTS   CPU   %CPU/R   %CPU/L   MEM     %MEM/R   %MEM/L IP               NODE                            AGE 
tp4-dp2-inference-pods-0                           *   1/1    Running               0   4020      n/a      n/a   57288      n/a      n/a 10.34.247.65     hyperpod-i-065b3eff5a05be7fc    29m
tp4-dp2-inference-pods-1                           *   1/1    Running               0   4023      n/a      n/a   59202      n/a      n/a 10.34.217.132    hyperpod-i-0de53e0b595777496    29m
```


### Summary

| Check | Status | Evidence |
|-------|--------|----------|
| TP=4 per node | OK | `nRanks 4 nNodes 1 localRanks 4` in TP comm splits; channels `0 1 2 3` |
| DP=2 across nodes | OK | `ncclCommSplit nranks 2 nNodes 2` for each of 4 DP pairs |
| EFA active | OK | `provider is efa`, `fabric is efa-direct`, 4 NICs, RDMA transport |
| aws-ofi-nccl | OK | `aws-ofi-nccl 1.16.2`, tuner for `p6-b200.48xlarge` |
| Libfabric | OK | `Libfabric (v10)`, version 2.1 |
| NVLink for TP | OK | `NVLS multicast` on all 4 devs, `P2P/CUMEM` on all intra-node channels |
| No cross-node TP | OK | TP channels contain only local ranks; EFA used only for DP comm splits |

### 1. TP=4, DP=2 topology

Rendezvous confirms 8 ranks across 2 nodes, 4 local ranks per pod:

```
[default] Rendezvous complete for workers. Result:
  restart_count=0
  master_addr=10.34.247.65
  master_port=1234
  group_rank=0
  group_world_size=8
  local_ranks=[0, 1, 2, 3]
  role_ranks=[0, 1, 2, 3]
  global_ranks=[0, 1, 2, 3]
  role_world_sizes=[8, 8, 8, 8]
  global_world_sizes=[8, 8, 8, 8]
```

NCCL version and CUDA driver on each rank:

```
[default0]:tp4-dp2-inference-pods-0:78:78 [0] NCCL INFO cudaDriverVersion 13000
[default0]:tp4-dp2-inference-pods-0:78:78 [0] NCCL INFO NCCL version 2.27.3+cuda12.9
```

Global communicator sees the full 8-rank, 2-node world:

```
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO comm 0x55bde5d092d0 rank 0 nRanks 8 nNodes 2 localRanks 4 localRank 0 MNNVL 0
[default1]:tp4-dp2-inference-pods-0:79:111 [1] NCCL INFO comm 0x5584f310dad0 rank 1 nRanks 8 nNodes 2 localRanks 4 localRank 1 MNNVL 0
[default2]:tp4-dp2-inference-pods-0:80:109 [2] NCCL INFO comm 0x564a232595b0 rank 2 nRanks 8 nNodes 2 localRanks 4 localRank 2 MNNVL 0
[default3]:tp4-dp2-inference-pods-0:81:110 [3] NCCL INFO comm 0x557cb94049d0 rank 3 nRanks 8 nNodes 2 localRanks 4 localRank 3 MNNVL 0
```

Global communicator init completes with timing breakdown:

```
[default0]:...:78:108 [0] NCCL INFO ncclCommInitRankConfig comm 0x55bde5d092d0 rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId 75000 commId 0xb4199a82c0d53582 - Init COMPLETE
[default0]:...:78:108 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 8 total 1.99 (kernels 0.25, alloc 0.19, bootstrap 0.42, allgathers 0.09, topo 0.52, graphs 0.29, connections 0.23, rest 0.00)
```

CommSplit creates 4 separate 2-rank DP sub-communicators (one per GPU index),
each spanning the two nodes. Each has a unique `color` and uses Libfabric (EFA):

```
[default0]:...:78:135 [0] NCCL INFO ncclCommSplit comm 0x55bde656e480 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 75000 parent 0x55bde5d092d0 splitCount 1 color 923304772 key 0- Init START
[default1]:...:79:129 [1] NCCL INFO ncclCommSplit comm 0x5584f34339f0 rank 0 nranks 2 cudaDev 1 nvmlDev 1 busId 76000 parent 0x5584f310dad0 splitCount 1 color 1449462471 key 0- Init START
[default2]:...:80:132 [2] NCCL INFO ncclCommSplit comm 0x564a2357f850 rank 0 nranks 2 cudaDev 2 nvmlDev 2 busId 86000 parent 0x564a232595b0 splitCount 1 color 120142604 key 0- Init START
[default3]:...:81:126 [3] NCCL INFO ncclCommSplit comm 0x557cb972b110 rank 0 nranks 2 cudaDev 3 nvmlDev 3 busId 87000 parent 0x557cb94049d0 splitCount 1 color 1211335945 key 0- Init START
```

CommSplit also creates the 4-rank TP group, entirely within a single node
(`nRanks 4 nNodes 1 localRanks 4`):

```
[default0]:tp4-dp2-inference-pods-0:78:140 [0] NCCL INFO comm 0x55bde6678070 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
[default3]:...:81:150 [3] NCCL INFO ncclCommSplit comm 0x557cb9834e50 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 87000 parent 0x557cb94049d0 splitCount 2 color 2003953581 key 3- Init START
```

### 2. EFA + aws-ofi-nccl + Libfabric

NCCL loads the `libnccl-net.so` plugin which exposes the Libfabric network:

```
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO Successfully loaded external plugin libnccl-net.so
```

The OFI plugin initializes with EFA as the provider, using RDMA transport:

```
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.16.2
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Using Libfabric version 2.1
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 12090
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Configuring AWS-specific options
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Setting provider_filter to efa
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Internode latency set at 35.0 us
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Using transport protocol RDMA (platform set)
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 4 nics)
```

4 EFA NICs detected per node at specific PCIe addresses:

```
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI NIC group 0 device #0 0000:71:00.0
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI NIC group 1 device #0 0000:72:00.0
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI NIC group 2 device #0 0000:84:00.0
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI NIC group 3 device #0 0000:85:00.0
```

DMA-BUF (GPU Direct RDMA) is available on all GPUs:

```
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Support for DMA-BUF registrations: true
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO DMA-BUF is available on GPU device 0
[default1]:tp4-dp2-inference-pods-0:79:129 [1] NCCL INFO DMA-BUF is available on GPU device 1
[default2]:tp4-dp2-inference-pods-0:80:132 [2] NCCL INFO DMA-BUF is available on GPU device 2
[default3]:tp4-dp2-inference-pods-0:81:126 [3] NCCL INFO DMA-BUF is available on GPU device 3
```

The OFI tuner is platform-aware and selects the optimal configuration for p6:

```
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Region base Tuner is chosen for platform: p6-b200.48xlarge
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Region Tuner init (platform 2): comm with 8 ranks and 2 nodes.
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NET/OFI Tuner init: comm with 8 ranks and 2 nodes.
```

Channel and buffer configuration auto-set by OFI plugin:

```
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO 32 coll channels, 32 collnet channels, 32 nvls channels, 32 p2p channels, 2 p2p channels per peer
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NCCL_BUFFSIZE set by environment to 8388608.
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NCCL_P2P_NET_CHUNKSIZE set by environment to 524288.
```

### 3. NVLink for intra-node TP

NVLS (NVLink Switch) multicast is available on all 4 GPUs. 32 channels for
the global communicator, 24 for the TP sub-communicator:

```
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NVLS multicast support is available on dev 0 (NVLS_NCHANNELS 32)
[default1]:tp4-dp2-inference-pods-0:79:111 [1] NCCL INFO NVLS multicast support is available on dev 1 (NVLS_NCHANNELS 32)
[default2]:tp4-dp2-inference-pods-0:80:109 [2] NCCL INFO NVLS multicast support is available on dev 2 (NVLS_NCHANNELS 32)
[default3]:tp4-dp2-inference-pods-0:81:110 [3] NCCL INFO NVLS multicast support is available on dev 3 (NVLS_NCHANNELS 32)
```

NVLS head mapping shows which ranks share NVLink across nodes (0<->4, 1<->5, etc.):

```
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NVLS Head  0:  0  4
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NVLS Head  1:  1  5
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NVLS Head  2:  2  6
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO NVLS Head  3:  3  7
```

Direct P2P is confirmed for the global communicator (intra-node ranks):

```
[default0]:tp4-dp2-inference-pods-0:78:108 [0] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0
```

TP sub-communicator channels only contain local ranks (0-3), confirming no
cross-node TP traffic:

```
[default0]:tp4-dp2-inference-pods-0:78:140 [0] NCCL INFO Channel 00/32 : 0 1 2 3
[default0]:tp4-dp2-inference-pods-0:78:140 [0] NCCL INFO Channel 01/32 : 0 1 2 3
[default0]:tp4-dp2-inference-pods-0:78:140 [0] NCCL INFO Channel 02/32 : 0 1 2 3
...
[default0]:tp4-dp2-inference-pods-0:78:140 [0] NCCL INFO Channel 31/32 : 0 1 2 3
```

All intra-node P2P connections use CUMEM (CUDA memory-mapped NVLink).
32 channels are established for each GPU pair (0->1, 0->2, 0->3):

```
[default0]:tp4-dp2-inference-pods-0:78:187 [0] NCCL INFO Channel 00/1 : 0[0] -> 1[1] via P2P/CUMEM
[default0]:tp4-dp2-inference-pods-0:78:187 [0] NCCL INFO Channel 01/1 : 0[0] -> 1[1] via P2P/CUMEM
...
[default0]:tp4-dp2-inference-pods-0:78:187 [0] NCCL INFO Channel 31/1 : 0[0] -> 1[1] via P2P/CUMEM
[default0]:tp4-dp2-inference-pods-0:78:187 [0] NCCL INFO Channel 00/1 : 0[0] -> 2[2] via P2P/CUMEM
...
[default0]:tp4-dp2-inference-pods-0:78:187 [0] NCCL INFO Channel 31/1 : 0[0] -> 2[2] via P2P/CUMEM
[default0]:tp4-dp2-inference-pods-0:78:187 [0] NCCL INFO Channel 00/1 : 0[0] -> 3[3] via P2P/CUMEM
...
[default0]:tp4-dp2-inference-pods-0:78:187 [0] NCCL INFO Channel 31/1 : 0[0] -> 3[3] via P2P/CUMEM
```

### 4. Minor warning (non-critical)

```
[default0]:libfabric:78:1775102137::core:core:cuda_gdrcopy_hmem_init():201<warn> gdr_open failed!
[default0]:libfabric:78:1775102137::core:core:cuda_hmem_init():816<warn> gdrcopy initialization failed! gdrcopy will not be used.
```

The Libfabric GDRCopy path is unavailable because the `gdrdrv` kernel module
is not loaded on the node. GDRCopy requires this host-side kernel module
(`/dev/gdrdrv`) to enable userspace GPU memory mapping - see
[NVIDIA/gdrcopy#251](https://github.com/NVIDIA/gdrcopy/issues/251). The
container ships the GDRCopy userspace library (`libgdrapi.so`) but the kernel
module must be installed on the host OS, which is outside the container's
control.

This does not affect performance because NCCL's DMA-BUF path is working
(confirmed by `DMA-BUF is available on GPU device N` on all GPUs). DMA-BUF
is a kernel-native mechanism for GPU Direct RDMA that does not require an
out-of-tree kernel module, unlike GDRCopy which depends on `gdrdrv`.

## Production: Argo Workflows for model download + inference

In the manual workflow above, the operator runs two commands sequentially:
`run_model_download.sh` then `run_hpto_job.sh`. In production, this should
be automated with a workflow engine that enforces ordering and handles
failures. Argo Workflows is a natural fit since it runs natively on
Kubernetes and can orchestrate arbitrary K8s resources.

The pattern is a two-step DAG:

```
  download-model  -->  run-inference
  (K8s Job)            (HyperPodPyTorchJob)
```

Step 1 runs the model download Job and blocks until completion. Step 2
only starts after step 1 succeeds.

### Example Argo Workflow

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: tp-inference-
  namespace: mlp
spec:
  entrypoint: download-and-infer
  arguments:
    parameters:
      - name: model-id
        value: "Qwen/Qwen2.5-72B-Instruct"
      - name: node-type
        value: "p6"
      - name: tp-size
        value: "4"
      - name: num-nodes
        value: "2"
      - name: gpus-per-node
        value: "4"
      - name: image-uri
        value: "<REGISTRY>/<USER>/dist-train/experiments:latest"

  templates:
    - name: download-and-infer
      dag:
        tasks:
          - name: download-model
            template: model-download
          - name: run-inference
            template: inference
            dependencies: [download-model]

    - name: model-download
      resource:
        action: create
        successCondition: status.succeeded > 0
        failureCondition: status.failed > 2
        manifest: |
          apiVersion: batch/v1
          kind: Job
          metadata:
            generateName: model-dl-
          spec:
            backoffLimit: 2
            ttlSecondsAfterFinished: 120
            template:
              spec:
                restartPolicy: OnFailure
                nodeSelector:
                  beta.kubernetes.io/instance-type: ml.{{workflow.parameters.node-type}}-b200.48xlarge
                tolerations:
                  - key: "nvidia.com/gpu"
                    operator: "Exists"
                    effect: "NoSchedule"
                volumes:
                  - name: fsx-static
                    persistentVolumeClaim:
                      claimName: fsx-static-claim
                containers:
                  - name: downloader
                    image: "{{workflow.parameters.image-uri}}"
                    resources:
                      requests:
                        nvidia.com/gpu: "1"
                      limits:
                        nvidia.com/gpu: "1"
                    env:
                      - name: HF_HOME
                        value: /mnt/fsx/asaha/hf_cache
                      - name: HF_HUB_DISABLE_XET
                        value: "1"
                    command: ["python", "-u", "-c"]
                    args:
                      - |
                        from huggingface_hub import snapshot_download
                        import time
                        t0 = time.time()
                        path = snapshot_download("{{workflow.parameters.model-id}}")
                        print(f"Done in {time.time()-t0:.1f}s -> {path}")
                    volumeMounts:
                      - name: fsx-static
                        mountPath: /mnt/fsx

    - name: inference
      resource:
        action: create
        successCondition: status.conditions.type == Completed
        failureCondition: status.conditions.type == Failed
        manifest: |
          apiVersion: sagemaker.amazonaws.com/v1
          kind: HyperPodPyTorchJob
          metadata:
            generateName: tp-infer-
          spec:
            nprocPerNode: "{{workflow.parameters.gpus-per-node}}"
            replicaSpecs:
              - name: pods
                replicas: {{workflow.parameters.num-nodes}}
                template:
                  spec:
                    # (same pod spec as hpto_tp.j2.yaml)
                    # nodeSelector, tolerations, volumes, env, etc.
                    containers:
                      - name: pytorch
                        image: "{{workflow.parameters.image-uri}}"
                        command:
                          - hyperpodrun
                          - "--nnodes={{workflow.parameters.num-nodes}}"
                          - "--nproc_per_node={{workflow.parameters.gpus-per-node}}"
                          - "/workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/tp_inference_72b.py"
                          - "--tp-size"
                          - "{{workflow.parameters.tp-size}}"
                          - "--output-dir"
                          - "/mnt/fsx/asaha/parallelism-experiments/results"
```

### Key points

- **`dependencies: [download-model]`** - Argo will not start inference until
  the download Job reports `status.succeeded > 0`.
- **`successCondition` / `failureCondition`** - Argo polls the resource status
  to determine step outcome. For standard K8s Jobs this is
  `status.succeeded > 0`. For HyperPodPyTorchJob, check the CRD's status
  conditions (adjust the condition type to match your CRD version).
- **`ttlSecondsAfterFinished: 120`** - Download job auto-cleans after 2 min,
  keeping the cluster tidy.
- **Parameterized** - Model ID, node type, TP size, and GPU count are all
  workflow parameters, so the same template works for different configurations.
- **Retry** - The download step has `backoffLimit: 2` for transient network
  failures. The inference step can similarly be wrapped with Argo's
  `retryStrategy` if desired.

### Submitting the workflow

```bash
# Submit directly
argo submit tp-inference-workflow.yaml -n mlp

# Submit with parameter overrides
argo submit tp-inference-workflow.yaml -n mlp \
  -p model-id=meta-llama/Llama-3.1-70B-Instruct \
  -p tp-size=8 \
  -p num-nodes=1

# Watch progress
argo watch @latest -n mlp

# List workflows
argo list -n mlp
```
