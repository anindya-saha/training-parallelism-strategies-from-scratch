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
TEMPLATE_FILE="${PROJECT_DIR}/k8s/hpto_job.yaml.template"
DRY_RUN=false
SKIP_BUILD=false

# -- Defaults -----------------------------------------------------------------
# Load your env.dev (REGISTRY, NAMESPACE, HF_TOKEN, etc.) before launching
# this script so the variables below pick up your overrides.

REGISTRY="${REGISTRY:?Set REGISTRY in your env.dev}"
EXPERIMENT_REPO="${REGISTRY}/${CURRENT_USER}/dist-train/experiments"

NAMESPACE="${NAMESPACE:-mlp}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-mlp-sa}"
FSX_CLAIM="${FSX_CLAIM:-fsx-static-claim}"
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

Launch distributed training experiments as a HyperPodPyTorchJob on Kubernetes.

Required:
  --job-name NAME           Job name (used as metadata.name)
  --node-type TYPE          Node type preset: p4d, p4de, p5en, p6
                            (or a full instance type like ml.p5en.48xlarge)
  --train-script PATH       Script path inside the container (e.g.
                            /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py)

Optional:
  --num-nodes NUM           Number of nodes / pods (default: 1)
  --gpus-per-node NUM       GPUs per node, also sets nproc_per_node (default: 8)
                            EFA adapters are always set to the node's full
                            allocation (e.g. 16 for p5en, 4 for p4d)

  --namespace NS            Kubernetes namespace (default: mlp)
  --service-account SA      Service account (default: mlp-sa)

  --model-id ID             (unused) Kept for backward compatibility.
                            Use run_model_download.sh instead.
  --train-args "ARGS"       Extra arguments passed to the train script
                            (e.g. "--tp-size 4 --max-new-tokens 64")

  --skip-build              Skip Docker build (reuse existing image)
  --image-uri URI           Use a specific image URI (implies --skip-build)
  --dry-run                 Render YAML to stdout without submitting

  -h, --help                Show this help

Examples:
  # Smoke test: run test_model.py on 1 p5en node
  $0 --job-name tp-smoke --node-type p5en \\
     --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py

  # TP training
  $0 --job-name tp-train --node-type p5en \\
     --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/train_tp.py

  # Multi-node TP + FSDP
  $0 --job-name tp-fsdp --node-type p5en --num-nodes 2 \\
     --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/train_tp_fsdp.py

  # Preview YAML without submitting
  $0 --job-name preview --node-type p5en --dry-run \\
     --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py

  # Reuse an already-pushed image
  $0 --job-name tp-test --node-type p5en --skip-build \\
     --train-script /workspace/training-parallelism-strategies-from-scratch/tensor-parallelism/test_model.py
EOF
  exit 1
}

# -- Parse args ---------------------------------------------------------------

TRAIN_SCRIPT=""
TRAIN_ARGS=""
MODEL_ID=""
IMAGE_URI=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --job-name)         JOB_NAME="$2"; shift 2 ;;
    --node-type)        NODE_TYPE_INPUT="$2"; shift 2 ;;
    --train-script)     TRAIN_SCRIPT="$2"; shift 2 ;;
    --train-args)       TRAIN_ARGS="$2"; shift 2 ;;
    --model-id)         MODEL_ID="$2"; shift 2 ;;
    --num-nodes)        NUM_NODES="$2"; shift 2 ;;
    --gpus-per-node)    GPU_PER_NODE="$2"; shift 2 ;;
    --namespace)        NAMESPACE="$2"; shift 2 ;;
    --service-account)  SERVICE_ACCOUNT="$2"; shift 2 ;;
    --image-uri)        IMAGE_URI="$2"; SKIP_BUILD=true; shift 2 ;;
    --skip-build)       SKIP_BUILD=true; shift ;;
    --dry-run)          DRY_RUN=true; shift ;;
    -h|--help)          usage ;;
    *)                  echo "Unknown option: $1"; usage ;;
  esac
done

# -- Validate -----------------------------------------------------------------

if [[ -z ${JOB_NAME:-} ]]; then
  echo "Error: --job-name is required"
  usage
fi

if [[ -z ${NODE_TYPE_INPUT:-} ]]; then
  echo "Error: --node-type is required"
  usage
fi

if [[ -z "${TRAIN_SCRIPT}" ]]; then
  echo "Error: --train-script is required"
  usage
fi

INSTANCE_TYPE="${NODE_PRESETS[$NODE_TYPE_INPUT]:-}"
if [[ -z "${INSTANCE_TYPE}" ]]; then
  echo "Error: unknown node type '${NODE_TYPE_INPUT}'. Valid presets: ${!NODE_PRESETS[*]}"
  exit 1
fi

GPU_TOTAL="${GPU_TOTAL_PRESETS[$NODE_TYPE_INPUT]}"
EFA_TOTAL="${EFA_TOTAL_PRESETS[$NODE_TYPE_INPUT]}"
EFA_PER_NODE=$(( EFA_TOTAL * GPU_PER_NODE / GPU_TOTAL ))

echo "==> Config: ${INSTANCE_TYPE}, ${GPU_PER_NODE} GPUs/node, ${EFA_PER_NODE} EFA/node, ${NUM_NODES} node(s)"

# -- Docker build & push ------------------------------------------------------

if [[ -n "${IMAGE_URI}" ]]; then
  echo "==> Using provided image: ${IMAGE_URI}"
elif [[ "${SKIP_BUILD}" == "false" ]]; then
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
  "JOB_NAME": "${JOB_NAME}",
  "NAMESPACE": "${NAMESPACE}",
  "SERVICE_ACCOUNT": "${SERVICE_ACCOUNT}",
  "INSTANCE_TYPE": "${INSTANCE_TYPE}",
  "IMAGE_URI": "${IMAGE_URI}",
  "TRAIN_SCRIPT": "${TRAIN_SCRIPT}",
  "TRAIN_ARGS": "${TRAIN_ARGS}",
  "MODEL_ID": "${MODEL_ID}",
  "NUM_NODES": ${NUM_NODES},
  "GPU_PER_NODE": "${GPU_PER_NODE}",
  "EFA_PER_NODE": "${EFA_PER_NODE}",
  "FSX_CLAIM": "${FSX_CLAIM}",
  "HF_CACHE_DIR": "${HF_CACHE_DIR:-/mnt/fsx/asaha/hf_cache}",
  "HF_TOKEN": "${HF_TOKEN:-}"
}
EOF

# -- Render template -----------------------------------------------------------

YAML_OUTPUT_FILE="${PWD}/${JOB_NAME}.yaml"

jinja2 "${TEMPLATE_FILE}" <(echo "${CONTEXT_JSON}") --format=json > "${YAML_OUTPUT_FILE}"
echo "==> Generated YAML: ${YAML_OUTPUT_FILE}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "---"
  cat "${YAML_OUTPUT_FILE}"
  exit 0
fi

# -- Submit to K8s -------------------------------------------------------------

RESOURCE_TYPE="hyperpodpytorchjob"

if kubectl get "${RESOURCE_TYPE}" "${JOB_NAME}" -n "${NAMESPACE}" &>/dev/null; then
  echo "==> HyperPodPyTorchJob '${JOB_NAME}' already exists. Deleting..."
  kubectl delete "${RESOURCE_TYPE}" "${JOB_NAME}" -n "${NAMESPACE}" --wait=true
fi

echo "==> Submitting HyperPodPyTorchJob..."
kubectl create -f "${YAML_OUTPUT_FILE}"
echo "==> Job '${JOB_NAME}' submitted to namespace '${NAMESPACE}'"

echo ""
echo "==> Useful commands"
echo "    Status:    kubectl get ${RESOURCE_TYPE} ${JOB_NAME} -n ${NAMESPACE}"
echo "    Describe:  kubectl describe ${RESOURCE_TYPE} ${JOB_NAME} -n ${NAMESPACE}"
echo "    Pods:      kubectl get pods -n ${NAMESPACE} -l job-name=${JOB_NAME}"
echo "    Shell:     kubectl exec -it -n ${NAMESPACE} \$(kubectl get pods -n ${NAMESPACE} -l job-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}') -c pytorch -- /bin/bash"
echo "    Logs:      kubectl logs -f -l job-name=${JOB_NAME} -n ${NAMESPACE} -c pytorch"
echo "    Delete:    kubectl delete ${RESOURCE_TYPE} ${JOB_NAME} -n ${NAMESPACE}"
