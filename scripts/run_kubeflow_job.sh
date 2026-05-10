#!/bin/bash
set -euo pipefail

CURRENT_USER=${USER:-${USERNAME:-${LOGNAME}}}
if [ -z "${CURRENT_USER+x}" ]; then
  echo "Error: unable to determine username. Set USER, USERNAME, or LOGNAME."
  exit 1
fi

if ! command -v jinja2 &>/dev/null; then
  echo "Error: jinja2 CLI not found. Install with: pip install jinja2-cli"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_TEMPLATE="${PROJECT_DIR}/k8s/kubeflow_runtime.yaml.template"
TRAINJOB_TEMPLATE="${PROJECT_DIR}/k8s/kubeflow_trainjob.yaml.template"
DRY_RUN=false
SKIP_BUILD=false
APPLY_RUNTIME=false
RUNTIME_ONLY=false

# -- Defaults -----------------------------------------------------------------

REGISTRY="${REGISTRY:?Set REGISTRY in your env.dev}"
EXPERIMENT_REPO="${REGISTRY}/${CURRENT_USER}/dist-train/experiments"

NAMESPACE="${NAMESPACE:-mlp}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-mlp-sa}"
FSX_CLAIM="${FSX_CLAIM:-fsx-static-claim}"
RUNTIME_NAME=""
NUM_NODES=1
GPU_PER_NODE=8

declare -A NODE_PRESETS=(
  [p4d]="ml.p4d.24xlarge"
  [p4de]="ml.p4de.24xlarge"
  [p5en]="ml.p5en.48xlarge"
  [p6]="ml.p6-b200.48xlarge"
)

declare -A GPU_TOTAL_PRESETS=(
  [p4d]=8
  [p4de]=8
  [p5en]=8
  [p6]=8
)

declare -A EFA_TOTAL_PRESETS=(
  [p4d]=4
  [p4de]=4
  [p5en]=16
  [p6]=8
)

# -- Usage --------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Launch distributed training experiments as a Kubeflow TrainJob on Kubernetes.

Required:
  --job-name NAME           Job name (used as metadata.name; not required with --runtime-only)
  --node-type TYPE          Node type preset: p4d, p4de, p5en, p6
                            (or a full instance type like ml.p5en.48xlarge)
  --train-script PATH       Script path inside the container (not required with --runtime-only)
                            (e.g. /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py)

Optional:
  --num-nodes NUM           Number of nodes / pods (default: 1)
  --gpus-per-node NUM       GPUs per node, also sets nproc_per_node (default: 8)
                            EFA adapters are always set to the node's full
                            allocation (e.g. 16 for p5en, 4 for p4d)

  --namespace NS            Kubernetes namespace (default: mlp)
  --service-account SA      Service account (default: mlp-sa)
  --runtime-name NAME       TrainingRuntime name
                            (default: torch-distributed-<node-type>, e.g. torch-distributed-p5en)

  --train-args "ARGS"       Extra arguments passed to the train script
                            (e.g. "--tp-size 4 --max-new-tokens 64")

  --apply-runtime           Also render and apply the TrainingRuntime
                            (use when runtime config has changed)
  --runtime-only            Deploy the TrainingRuntime only (skip TrainJob).
                            Useful for initial cluster setup or runtime updates.

  --skip-build              Skip Docker build (reuse existing image)
  --image-uri URI           Use a specific image URI (implies --skip-build)
  --dry-run                 Render YAML to stdout without submitting

  -h, --help                Show this help

Examples:
  # Smoke test: run test_model.py on 1 p5en node
  $0 --job-name tp-smoke --node-type p5en \\
     --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py

  # With training args
  $0 --job-name cp-train --node-type p5en \\
     --train-script /workspace/training-parallelism-strategies-from-scratch/context-parallelism/src/train_gpt_cp.py \\
     --train-args "--config medium --seq-len 1024 --cp-size 8"

  # Multi-node with runtime apply
  $0 --job-name tp-fsdp --node-type p5en --num-nodes 2 --apply-runtime \\
     --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/train_tp_fsdp.py

  # Preview YAML without submitting
  $0 --job-name preview --node-type p5en --dry-run \\
     --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py

  # Reuse an already-pushed image
  $0 --job-name tp-test --node-type p5en --skip-build \\
     --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py

  # Deploy only the TrainingRuntime (no TrainJob)
  $0 --node-type p5en --runtime-only --image-uri <your-image>
EOF
  exit 1
}

# -- Parse args ---------------------------------------------------------------

TRAIN_SCRIPT=""
TRAIN_ARGS=""
IMAGE_URI=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --job-name)         JOB_NAME="$2"; shift 2 ;;
    --node-type)        NODE_TYPE_INPUT="$2"; shift 2 ;;
    --train-script)     TRAIN_SCRIPT="$2"; shift 2 ;;
    --train-args)       TRAIN_ARGS="$2"; shift 2 ;;
    --num-nodes)        NUM_NODES="$2"; shift 2 ;;
    --gpus-per-node)    GPU_PER_NODE="$2"; shift 2 ;;
    --namespace)        NAMESPACE="$2"; shift 2 ;;
    --service-account)  SERVICE_ACCOUNT="$2"; shift 2 ;;
    --runtime-name)     RUNTIME_NAME="$2"; shift 2 ;;
    --image-uri)        IMAGE_URI="$2"; SKIP_BUILD=true; shift 2 ;;
    --skip-build)       SKIP_BUILD=true; shift ;;
    --apply-runtime)    APPLY_RUNTIME=true; shift ;;
    --runtime-only)     RUNTIME_ONLY=true; APPLY_RUNTIME=true; shift ;;
    --dry-run)          DRY_RUN=true; shift ;;
    -h|--help)          usage ;;
    *)                  echo "Unknown option: $1"; usage ;;
  esac
done

# -- Validate -----------------------------------------------------------------

if [[ -z ${NODE_TYPE_INPUT:-} ]]; then
  echo "Error: --node-type is required"
  usage
fi

if [[ "${RUNTIME_ONLY}" == "false" ]]; then
  if [[ -z ${JOB_NAME:-} ]]; then
    echo "Error: --job-name is required"
    usage
  fi

  if [[ -z "${TRAIN_SCRIPT}" ]]; then
    echo "Error: --train-script is required"
    usage
  fi
fi

INSTANCE_TYPE="${NODE_PRESETS[$NODE_TYPE_INPUT]:-}"
if [[ -z "${INSTANCE_TYPE}" ]]; then
  echo "Error: unknown node type '${NODE_TYPE_INPUT}'. Valid presets: ${!NODE_PRESETS[*]}"
  exit 1
fi

if [[ -z "${RUNTIME_NAME}" ]]; then
  RUNTIME_NAME="torch-distributed-${NODE_TYPE_INPUT}"
fi

GPU_TOTAL="${GPU_TOTAL_PRESETS[$NODE_TYPE_INPUT]}"
EFA_TOTAL="${EFA_TOTAL_PRESETS[$NODE_TYPE_INPUT]}"
EFA_PER_NODE=$(( EFA_TOTAL * GPU_PER_NODE / GPU_TOTAL ))

echo "==> Config: ${INSTANCE_TYPE}, ${GPU_PER_NODE} GPUs/node, ${EFA_PER_NODE} EFA/node, ${NUM_NODES} node(s)"
echo "==> Runtime: ${RUNTIME_NAME}"

# -- Docker build & push ------------------------------------------------------

if [[ -n "${IMAGE_URI}" ]]; then
  echo "==> Using provided image: ${IMAGE_URI}"
elif [[ "${SKIP_BUILD}" == "false" && "${RUNTIME_ONLY}" == "false" ]]; then
  eval "$("${SCRIPT_DIR}/build_image.sh")"
else
  IMAGE_URI=$(docker images "${EXPERIMENT_REPO}" --format '{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}' | sort -r | head -1 | cut -f2)
  if [[ -z "${IMAGE_URI}" ]]; then
    echo "Error: no local image found for ${EXPERIMENT_REPO}. Run without --skip-build first."
    exit 1
  fi
  echo "==> Reusing image: ${IMAGE_URI}"
fi

# -- Build JSON context -------------------------------------------------------

read -r -d '' CONTEXT_JSON <<EOF || true
{
  "JOB_NAME": "${JOB_NAME:-}",
  "NAMESPACE": "${NAMESPACE}",
  "SERVICE_ACCOUNT": "${SERVICE_ACCOUNT}",
  "INSTANCE_TYPE": "${INSTANCE_TYPE}",
  "IMAGE_URI": "${IMAGE_URI}",
  "TRAIN_SCRIPT": "${TRAIN_SCRIPT}",
  "TRAIN_ARGS": "${TRAIN_ARGS}",
  "RUNTIME_NAME": "${RUNTIME_NAME}",
  "NUM_NODES": ${NUM_NODES},
  "GPU_PER_NODE": "${GPU_PER_NODE}",
  "EFA_PER_NODE": "${EFA_PER_NODE}",
  "FSX_CLAIM": "${FSX_CLAIM}",
  "HF_CACHE_DIR": "${HF_CACHE_DIR:-/mnt/fsx/asaha/hf_cache}",
  "HF_TOKEN": "${HF_TOKEN:-}"
}
EOF

# -- Render and apply TrainingRuntime (if requested) ---------------------------

if [[ "${APPLY_RUNTIME}" == "true" ]]; then
  RUNTIME_YAML="${PWD}/${RUNTIME_NAME}-runtime.yaml"
  jinja2 "${RUNTIME_TEMPLATE}" <(echo "${CONTEXT_JSON}") --format=json > "${RUNTIME_YAML}"
  echo "==> Generated TrainingRuntime YAML: ${RUNTIME_YAML}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "--- TrainingRuntime ---"
    cat "${RUNTIME_YAML}"
  else
    kubectl apply -f "${RUNTIME_YAML}"
    echo "==> TrainingRuntime '${RUNTIME_NAME}' applied to namespace '${NAMESPACE}'"
  fi

  if [[ "${RUNTIME_ONLY}" == "true" ]]; then
    echo ""
    echo "==> Useful commands"
    echo "    Status:    kubectl get trainingruntime ${RUNTIME_NAME} -n ${NAMESPACE}"
    echo "    Describe:  kubectl describe trainingruntime ${RUNTIME_NAME} -n ${NAMESPACE}"
    echo "    Delete:    kubectl delete trainingruntime ${RUNTIME_NAME} -n ${NAMESPACE}"
    exit 0
  fi
fi

# -- Render TrainJob template --------------------------------------------------

TRAINJOB_YAML="${PWD}/${JOB_NAME}.yaml"
jinja2 "${TRAINJOB_TEMPLATE}" <(echo "${CONTEXT_JSON}") --format=json > "${TRAINJOB_YAML}"
echo "==> Generated TrainJob YAML: ${TRAINJOB_YAML}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "--- TrainJob ---"
  cat "${TRAINJOB_YAML}"
  exit 0
fi

# -- Submit to K8s -------------------------------------------------------------

RESOURCE_TYPE="trainjob.trainer.kubeflow.org"

if kubectl get "${RESOURCE_TYPE}" "${JOB_NAME}" -n "${NAMESPACE}" &>/dev/null; then
  echo "==> TrainJob '${JOB_NAME}' already exists. Deleting..."
  kubectl delete "${RESOURCE_TYPE}" "${JOB_NAME}" -n "${NAMESPACE}" --wait=true
fi

echo "==> Submitting TrainJob..."
kubectl create -f "${TRAINJOB_YAML}"
echo "==> TrainJob '${JOB_NAME}' submitted to namespace '${NAMESPACE}'"

echo ""
echo "==> Useful commands"
echo "    Status:    kubectl get ${RESOURCE_TYPE} ${JOB_NAME} -n ${NAMESPACE}"
echo "    Describe:  kubectl describe ${RESOURCE_TYPE} ${JOB_NAME} -n ${NAMESPACE}"
echo "    Pods:      kubectl get pods -n ${NAMESPACE} -l trainer.kubeflow.org/trainjob-name=${JOB_NAME}"
echo "    Shell:     kubectl exec -it -n ${NAMESPACE} \$(kubectl get pods -n ${NAMESPACE} -l trainer.kubeflow.org/trainjob-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}') -c node -- /bin/bash"
echo "    Logs:      kubectl logs -f -l trainer.kubeflow.org/trainjob-name=${JOB_NAME} -n ${NAMESPACE} -c node"
echo "    Delete:    kubectl delete ${RESOURCE_TYPE} ${JOB_NAME} -n ${NAMESPACE}"
