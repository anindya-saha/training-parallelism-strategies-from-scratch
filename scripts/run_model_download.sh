#!/bin/bash
set -euo pipefail

CURRENT_USER=${USER:-${USERNAME:-${LOGNAME}}}
if [ -z "${CURRENT_USER+x}" ]; then
  echo "Error: unable to determine username. Set USER, USERNAME, or LOGNAME."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE_FILE="${PROJECT_DIR}/k8s/model_download.j2.yaml"

if ! command -v jinja2 &>/dev/null; then
  echo "Error: jinja2 CLI not found. Install with: pip install jinja2-cli"
  exit 1
fi

# -- Defaults -----------------------------------------------------------------
# Load your dev.env (REGISTRY, NAMESPACE, HF_CACHE_DIR, HF_TOKEN, etc.)
# before launching this script so the variables below pick up your overrides.

REGISTRY="${REGISTRY:?Set REGISTRY in your dev.env}"
EXPERIMENT_REPO="${REGISTRY}/${CURRENT_USER}/dist-train/experiments"

NAMESPACE="${NAMESPACE:-mlp}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-mlp-sa}"
FSX_CLAIM="${FSX_CLAIM:-fsx-static-claim}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/fsx/asaha/hf_cache}"

DRY_RUN=false

# -- Usage --------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Download a HuggingFace model to shared FSx storage via a Kubernetes Job.

Required:
  --model-id ID             HuggingFace model ID (e.g. "Qwen/Qwen2.5-72B-Instruct")

Optional:
  --image-uri URI           Container image (default: latest local experiment image)
  --job-name NAME           Job name (default: model-download)
  --cache-dir PATH          HF cache directory on FSx (default: ${HF_CACHE_DIR})
  --namespace NS            Kubernetes namespace (default: ${NAMESPACE})
  --service-account SA      Service account (default: ${SERVICE_ACCOUNT})
  --dry-run                 Render YAML to stdout without submitting
  -h, --help                Show this help

Examples:
  # Download Qwen2.5-72B to FSx (uses latest local experiment image)
  $0 --model-id Qwen/Qwen2.5-72B-Instruct

  # Download a gated model (requires HF_TOKEN exported)
  $0 --model-id meta-llama/Llama-3.1-70B-Instruct

  # Use a specific image
  $0 --model-id Qwen/Qwen2.5-72B-Instruct \\
     --image-uri \${REGISTRY}/\${USER}/dist-train/experiments:abc123

  # Preview the YAML
  $0 --model-id Qwen/Qwen2.5-72B-Instruct --dry-run
EOF
  exit 1
}

# -- Parse args ---------------------------------------------------------------

JOB_NAME="model-download"
MODEL_ID=""
IMAGE_URI=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-id)         MODEL_ID="$2"; shift 2 ;;
    --image-uri)        IMAGE_URI="$2"; shift 2 ;;
    --job-name)         JOB_NAME="$2"; shift 2 ;;
    --cache-dir)        HF_CACHE_DIR="$2"; shift 2 ;;
    --namespace)        NAMESPACE="$2"; shift 2 ;;
    --service-account)  SERVICE_ACCOUNT="$2"; shift 2 ;;
    --dry-run)          DRY_RUN=true; shift ;;
    -h|--help)          usage ;;
    *)                  echo "Unknown option: $1"; usage ;;
  esac
done

# -- Validate -----------------------------------------------------------------

if [[ -z "${MODEL_ID}" ]]; then
  echo "Error: --model-id is required"
  usage
fi

# -- Resolve image URI --------------------------------------------------------

if [[ -n "${IMAGE_URI}" ]]; then
  echo "==> Using provided image: ${IMAGE_URI}"
else
  IMAGE_URI=$(docker images "${EXPERIMENT_REPO}" --format '{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}' | sort -r | head -1 | cut -f2)
  if [[ -z "${IMAGE_URI}" ]]; then
    echo "Error: no local image found for ${EXPERIMENT_REPO}. Pass --image-uri explicitly."
    exit 1
  fi
  echo "==> Using latest local image: ${IMAGE_URI}"
fi

echo "==> Downloading model: ${MODEL_ID}"
echo "==> Cache dir: ${HF_CACHE_DIR}"

# -- Build JSON context -------------------------------------------------------

read -r -d '' CONTEXT_JSON <<EOF || true
{
  "JOB_NAME": "${JOB_NAME}",
  "NAMESPACE": "${NAMESPACE}",
  "SERVICE_ACCOUNT": "${SERVICE_ACCOUNT}",
  "IMAGE_URI": "${IMAGE_URI}",
  "MODEL_ID": "${MODEL_ID}",
  "HF_CACHE_DIR": "${HF_CACHE_DIR}",
  "HF_TOKEN": "${HF_TOKEN:-}",
  "FSX_CLAIM": "${FSX_CLAIM}"
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

if kubectl get job "${JOB_NAME}" -n "${NAMESPACE}" &>/dev/null; then
  echo "==> Job '${JOB_NAME}' already exists. Deleting..."
  kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}" --wait=true
fi

echo "==> Submitting model download job..."
kubectl create -f "${YAML_OUTPUT_FILE}"
echo "==> Job '${JOB_NAME}' submitted to namespace '${NAMESPACE}'"

echo ""
echo "==> Useful commands:"
echo "    Logs:      kubectl logs -f job/${JOB_NAME} -n ${NAMESPACE}"
echo "    Status:    kubectl get job ${JOB_NAME} -n ${NAMESPACE}"
echo "    Delete:    kubectl delete job ${JOB_NAME} -n ${NAMESPACE}"

# -- Wait for completion ------------------------------------------------------

echo ""
echo "==> Waiting for job to complete (polling every 30s, timeout 30 min)..."
SECONDS=0
while (( SECONDS < 1800 )); do
  STATUS=$(kubectl get job "${JOB_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)
  if [[ "${STATUS}" == "True" ]]; then
    echo "==> Model download complete (${SECONDS}s elapsed)."
    exit 0
  fi
  FAILED=$(kubectl get job "${JOB_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)
  if [[ "${FAILED}" == "True" ]]; then
    echo "==> ERROR: job failed. Check logs:"
    echo "    kubectl logs -f job/${JOB_NAME} -n ${NAMESPACE}"
    exit 1
  fi
  PHASE=$(kubectl get pods -l job-name="${JOB_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
  echo "    Still downloading... K8s pod phase=${PHASE:-Pending} (${SECONDS}s elapsed)"
  sleep 30
done

echo "==> ERROR: timed out after 30 min"
exit 1
