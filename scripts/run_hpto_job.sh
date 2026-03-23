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
TEMPLATE_FILE="${PROJECT_DIR}/k8s/hpto_tp.j2.yaml"
DRY_RUN=false
SKIP_BUILD=false

# -- Defaults -----------------------------------------------------------------

REGISTRY=mlp.docker.acme.com
EXPERIMENT_REPO="${REGISTRY}/${CURRENT_USER}/dist-train/experiments"

NAMESPACE=mlp
SERVICE_ACCOUNT=mlp-sa
NPROC_PER_NODE=8
NUM_NODES=1
GPUS_PER_NODE=8
EFA_PER_NODE=""

declare -A NODE_PRESETS=(
  [p4d]="ml.p4d.24xlarge"
  [p4de]="ml.p4de.24xlarge"
  [p5en]="ml.p5en.48xlarge"
  [p6]="ml.p6-b200.48xlarge"
)

declare -A EFA_PRESETS=(
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
  --num-nodes NUM           Number of nodes (default: 1)
  --nproc-per-node NUM      Processes per node (default: 8)
  --gpus-per-node NUM       GPUs to request per node (default: 8)
  --efa-per-node NUM        EFA adapters to request (default: auto from node type)

  --namespace NS            Kubernetes namespace (default: mlp)
  --service-account SA      Service account (default: mlp-sa)

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
IMAGE_URI=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --job-name)         JOB_NAME="$2"; shift 2 ;;
    --node-type)        NODE_TYPE_INPUT="$2"; shift 2 ;;
    --train-script)     TRAIN_SCRIPT="$2"; shift 2 ;;
    --num-nodes)        NUM_NODES="$2"; shift 2 ;;
    --nproc-per-node)   NPROC_PER_NODE="$2"; shift 2 ;;
    --gpus-per-node)    GPUS_PER_NODE="$2"; shift 2 ;;
    --efa-per-node)     EFA_PER_NODE="$2"; shift 2 ;;
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

INSTANCE_TYPE="${NODE_PRESETS[$NODE_TYPE_INPUT]:-$NODE_TYPE_INPUT}"

if [[ -z "${EFA_PER_NODE}" ]]; then
  EFA_PER_NODE="${EFA_PRESETS[$NODE_TYPE_INPUT]:-16}"
fi

echo "==> Config: ${INSTANCE_TYPE}, ${GPUS_PER_NODE} GPUs, ${EFA_PER_NODE} EFA, ${NPROC_PER_NODE} procs/node, ${NUM_NODES} node(s)"

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
  "NUM_NODES": ${NUM_NODES},
  "NPROC_PER_NODE": "${NPROC_PER_NODE}",
  "GPUS_PER_NODE": "${GPUS_PER_NODE}",
  "EFA_PER_NODE": "${EFA_PER_NODE}"
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
echo "    Logs:      kubectl logs -f -l job-name=${JOB_NAME} -n ${NAMESPACE} --all-containers"
echo "    Delete:    kubectl delete ${RESOURCE_TYPE} ${JOB_NAME} -n ${NAMESPACE}"
