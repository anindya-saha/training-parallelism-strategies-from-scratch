#!/bin/bash
set -euo pipefail

CURRENT_USER=${USER:-${USERNAME:-${LOGNAME}}}
if [ -z "${CURRENT_USER+x}" ]; then
  echo "Error: unable to determine username. Set USER, USERNAME, or LOGNAME."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -- Defaults -----------------------------------------------------------------
# Load your dev.env (REGISTRY, etc.) before launching this script so the
# variables below pick up your overrides.

REGISTRY="${REGISTRY:?Set REGISTRY in your dev.env}"
PUSH=true

# -- Usage --------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Build and optionally push the thin experiment image for distributed training.

The image layers source files from tensor-parallelism/src/ (and future
parallelism strategies) on top of the heavy CUDA/PyTorch base image.

Options:
  --registry URL            Container registry
  --experiment-repo REPO    Image repository (default: <registry>/<user>/dist-train/experiments)
  --no-push                 Build only, do not push to registry
  -h, --help                Show this help

Outputs (written to stdout on the last line):
  IMAGE_URI=<image-uri>

Examples:
  # Build and push
  $0

  # Build only, no push
  $0 --no-push

  # Use in run_hpto_job.sh via eval
  eval \$($0)
EOF
  exit 1
}

# -- Parse args ---------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case $1 in
    --registry)          REGISTRY="$2"; shift 2 ;;
    --experiment-repo)   EXPERIMENT_REPO="$2"; shift 2 ;;
    --no-push)           PUSH=false; shift ;;
    -h|--help)           usage ;;
    *)                   echo "Unknown option: $1"; usage ;;
  esac
done

EXPERIMENT_REPO="${EXPERIMENT_REPO:-${REGISTRY}/${CURRENT_USER}/dist-train/experiments}"

# -- Build image --------------------------------------------------------------

echo "==> Building experiment image..." >&2
docker build -f "${PROJECT_DIR}/docker/Dockerfile.experiments" \
  -t "${EXPERIMENT_REPO}:temp" \
  "${PROJECT_DIR}" >&2

IMAGE_SHA=$(docker inspect --format='{{.Id}}' "${EXPERIMENT_REPO}:temp")
IMAGE_SHA="${IMAGE_SHA#sha256:}"
IMAGE_URI="${EXPERIMENT_REPO}:${IMAGE_SHA}"
docker tag "${EXPERIMENT_REPO}:temp" "${IMAGE_URI}" >&2
docker rmi "${EXPERIMENT_REPO}:temp" >/dev/null 2>&1 || true

# -- Push ----------------------------------------------------------------------

if [[ "${PUSH}" == "true" ]]; then
  echo "==> Pushing image..." >&2
  docker push "${IMAGE_URI}" >&2
fi

# -- Output image URI (machine-readable) --------------------------------------

echo "==> Experiment image: ${IMAGE_URI}" >&2
echo "IMAGE_URI=${IMAGE_URI}"
