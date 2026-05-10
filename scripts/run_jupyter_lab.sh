#!/bin/bash
set -euo pipefail

CURRENT_USER=${USER:-${USERNAME:-${LOGNAME}}}
if [ -z "${CURRENT_USER+x}" ]; then
  echo "Error: unable to determine username. Set USER, USERNAME, or LOGNAME."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE_FILE="${PROJECT_DIR}/k8s/jupyter_lab.yaml.template"

if ! command -v jinja2 &>/dev/null; then
  echo "Error: jinja2 CLI not found. Install with: uv sync (repo) or pip install jinja2-cli"
  exit 1
fi

# -- Defaults -----------------------------------------------------------------

REGISTRY="${REGISTRY:?Set REGISTRY in your env.dev}"
EXPERIMENT_REPO="${REGISTRY}/${CURRENT_USER}/dist-train/experiments"

NAMESPACE="${NAMESPACE:-mlp}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-mlp-sa}"
FSX_CLAIM="${FSX_CLAIM:-fsx-static-claim}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/fsx/asaha/hf_cache}"

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

declare -A CPU_TOTAL_PRESETS=(
  [p4d]=96
  [p4de]=96
  [p5en]=192
  [p6]=192
)

declare -A MEMORY_TOTAL_PRESETS=(
  [p4d]="1100Gi"
  [p4de]="1100Gi"
  [p5en]="2000Gi"
  [p6]="2000Gi"
)

DEPLOYMENT_NAME="jupyter-lab"
NODE_TYPE_INPUT=""
IMAGE_URI=""
DRY_RUN=false
SKIP_BUILD=false

# -- Usage --------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Render k8s/jupyter_lab.deployment.j2.yaml, delete existing Deployment/Service if any, then kubectl create.

Required:
  --node-type TYPE          Same presets as run_hpto_job.sh: p4d, p4de, p5en, p6
                            (or a full instance type string, e.g. ml.p5en.48xlarge)

Optional:
  --deployment-name NAME    Deployment + service prefix (default: ${DEPLOYMENT_NAME})
  --image-uri URI           Container image (implies --skip-build)
  --skip-build              Skip Docker build; reuse newest local ${EXPERIMENT_REPO} image
  --gpu-count N             nvidia.com/gpu limit and request (default: node's GPU_TOTAL_PRESETS)
  --namespace NS            Kubernetes namespace (default: ${NAMESPACE})
  --jupyter-token STR       Server token (default: generate with openssl and print once)
  --dry-run                 Render YAML to stdout only (no kubectl apply)

Examples:
  source dev.env
  # Build, push, deploy (default: runs build_image.sh)
  $0 --node-type p5en

  # Reuse last local image without rebuilding
  $0 --node-type p5en --skip-build

  $0 --node-type p5en --deployment-name jupyter-pp --gpu-count 8 --dry-run
EOF
  exit 1
}

# -- Parse args ---------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case $1 in
    --deployment-name)  DEPLOYMENT_NAME="$2"; shift 2 ;;
    --node-type)        NODE_TYPE_INPUT="$2"; shift 2 ;;
    --image-uri)        IMAGE_URI="$2"; SKIP_BUILD=true; shift 2 ;;
    --skip-build)       SKIP_BUILD=true; shift ;;
    --gpu-count)        GPU_COUNT="$2"; shift 2 ;;
    --namespace)        NAMESPACE="$2"; shift 2 ;;
    --jupyter-token)    JUPYTER_TOKEN="$2"; shift 2 ;;
    --dry-run)          DRY_RUN=true; shift ;;
    -h|--help)          usage ;;
    *)                  echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "${NODE_TYPE_INPUT}" ]]; then
  echo "Error: --node-type is required (preset: p4d, p4de, p5en, p6, or full instance type)"
  usage
fi

INSTANCE_TYPE="${NODE_PRESETS[$NODE_TYPE_INPUT]:-}"
if [[ -z "${INSTANCE_TYPE}" ]]; then
  if [[ "${NODE_TYPE_INPUT}" == ml.* ]]; then
    INSTANCE_TYPE="${NODE_TYPE_INPUT}"
  else
    echo "Error: unknown node type '${NODE_TYPE_INPUT}'. Valid presets: ${!NODE_PRESETS[*]}"
    echo "       Or pass a full instance type label (e.g. ml.p5en.48xlarge)."
    exit 1
  fi
fi

# Derive resource defaults from presets (flags/env override these)
GPU_TOTAL="${GPU_TOTAL_PRESETS[$NODE_TYPE_INPUT]:-8}"
CPU_TOTAL="${CPU_TOTAL_PRESETS[$NODE_TYPE_INPUT]:-192}"
MEMORY_TOTAL="${MEMORY_TOTAL_PRESETS[$NODE_TYPE_INPUT]:-2000Gi}"

# Request = 80% of total, Limit = 90% of total
MEMORY_NUM="${MEMORY_TOTAL%Gi}"
CPU_REQUEST_DEFAULT=$(( CPU_TOTAL * 80 / 100 ))
CPU_LIMIT_DEFAULT=$(( CPU_TOTAL * 90 / 100 ))
MEMORY_REQUEST_DEFAULT="$(( MEMORY_NUM * 80 / 100 ))Gi"
MEMORY_LIMIT_DEFAULT="$(( MEMORY_NUM * 90 / 100 ))Gi"

GPU_COUNT="${GPU_COUNT:-$GPU_TOTAL}"
CPU_REQUEST="${CPU_REQUEST:-$CPU_REQUEST_DEFAULT}"
CPU_LIMIT="${CPU_LIMIT:-$CPU_LIMIT_DEFAULT}"
MEMORY_REQUEST="${MEMORY_REQUEST:-$MEMORY_REQUEST_DEFAULT}"
MEMORY_LIMIT="${MEMORY_LIMIT:-$MEMORY_LIMIT_DEFAULT}"

echo "==> Instance type (nodeSelector): ${INSTANCE_TYPE}" >&2
echo "==> Resources: ${GPU_COUNT} GPUs, CPU ${CPU_REQUEST}-${CPU_LIMIT}, Memory ${MEMORY_REQUEST}-${MEMORY_LIMIT}" >&2

if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
  JUPYTER_TOKEN="$(openssl rand -hex 24)"
  echo "==> Generated JUPYTER_TOKEN (save this for the browser):" >&2
  echo "${JUPYTER_TOKEN}" >&2
  echo "" >&2
fi

# -- Docker build & push (same pattern as run_hpto_job.sh) ---------------------

if [[ -n "${IMAGE_URI}" ]]; then
  echo "==> Using provided image: ${IMAGE_URI}" >&2
elif [[ "${SKIP_BUILD}" == "false" ]]; then
  eval "$("${SCRIPT_DIR}/build_image.sh")"
else
  IMAGE_URI=$(docker images "${EXPERIMENT_REPO}" --format '{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}' | sort -r | head -1 | cut -f2)
  if [[ -z "${IMAGE_URI}" ]]; then
    echo "Error: no local image found for ${EXPERIMENT_REPO}. Run without --skip-build first."
    exit 1
  fi
  echo "==> Reusing image: ${IMAGE_URI}" >&2
fi

# -- Build JSON context -------------------------------------------------------

read -r -d '' CONTEXT_JSON <<EOF || true
{
  "DEPLOYMENT_NAME": "${DEPLOYMENT_NAME}",
  "NAMESPACE": "${NAMESPACE}",
  "SERVICE_ACCOUNT": "${SERVICE_ACCOUNT}",
  "IMAGE_URI": "${IMAGE_URI}",
  "JUPYTER_TOKEN": "${JUPYTER_TOKEN}",
  "HF_CACHE_DIR": "${HF_CACHE_DIR}",
  "HF_TOKEN": "${HF_TOKEN:-}",
  "FSX_CLAIM": "${FSX_CLAIM}",
  "INSTANCE_TYPE": "${INSTANCE_TYPE}",
  "GPU_COUNT": "${GPU_COUNT}",
  "CPU_REQUEST": "${CPU_REQUEST}",
  "CPU_LIMIT": "${CPU_LIMIT}",
  "MEMORY_REQUEST": "${MEMORY_REQUEST}",
  "MEMORY_LIMIT": "${MEMORY_LIMIT}"
}
EOF

# Same as run_hpto_job.sh: write rendered manifest to ${PWD}/<name>.yaml (run from repo root or your choice of cwd).
YAML_OUTPUT_FILE="${PWD}/${DEPLOYMENT_NAME}.yaml"

jinja2 "${TEMPLATE_FILE}" <(echo "${CONTEXT_JSON}") --format=json > "${YAML_OUTPUT_FILE}"
echo "==> Generated YAML: ${YAML_OUTPUT_FILE}" >&2

if [[ "${DRY_RUN}" == "true" ]]; then
  cat "${YAML_OUTPUT_FILE}"
  exit 0
fi

# Delete whatever this manifest defines (Deployment + Service) so kubectl create succeeds.
if [[ -f "${YAML_OUTPUT_FILE}" ]]; then
  echo "==> Deleting existing resources from ${YAML_OUTPUT_FILE} (if any)..." >&2
  kubectl delete -f "${YAML_OUTPUT_FILE}" --ignore-not-found --wait=true
fi

echo "==> Submitting Deployment and Service..." >&2
kubectl create -f "${YAML_OUTPUT_FILE}"

echo "" >&2
echo "==> Port-forward (run locally):" >&2
echo "    kubectl port-forward -n ${NAMESPACE} svc/${DEPLOYMENT_NAME}-svc 8888:8888" >&2
echo "" >&2
echo "==> Open: http://127.0.0.1:8888/lab?token=${JUPYTER_TOKEN}" >&2
