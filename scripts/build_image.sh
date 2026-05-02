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

REGISTRY="${REGISTRY:?Set REGISTRY in your env.dev}"
BASE_IMAGE="${BASE_IMAGE:-}"
PUSH=true

# -- Usage --------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Build and optionally push the thin experiment image for distributed training.

Layers experiment source files on top of the base PyTorch image built by
build_base_image.sh. Pass the base image via --base-image or BASE_IMAGE env var.

Options:
  --base-image URI          Base image URI (required, or set BASE_IMAGE env var)
  --registry URL            Container registry
  --experiment-repo REPO    Image repository (default: <registry>/<user>/dist-train/experiments)
  --no-push                 Build only, do not push to registry
  -h, --help                Show this help

Outputs (written to stdout on the last line):
  IMAGE_URI=<image-uri>

Examples:
  # Build base first, then experiments
  eval \$(./scripts/build_base_image.sh --no-push)
  $0 --base-image \$BASE_IMAGE_URI --no-push

  # Or pass BASE_IMAGE directly
  BASE_IMAGE=pytorch-base:runtime $0 --no-push
EOF
  exit 1
}

# -- Parse args ---------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case $1 in
    --base-image)        BASE_IMAGE="$2"; shift 2 ;;
    --registry)          REGISTRY="$2"; shift 2 ;;
    --experiment-repo)   EXPERIMENT_REPO="$2"; shift 2 ;;
    --no-push)           PUSH=false; shift ;;
    -h|--help)           usage ;;
    *)                   echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "${BASE_IMAGE}" ]]; then
  echo "Error: --base-image is required (or set BASE_IMAGE env var)." >&2
  echo "       Build the base image first: ./scripts/build_base_image.sh" >&2
  exit 1
fi

EXPERIMENT_REPO="${EXPERIMENT_REPO:-${REGISTRY}/${CURRENT_USER}/dist-train/experiments}"

# -- Build image --------------------------------------------------------------

echo "==> Building experiment image..." >&2
echo "    Base image: ${BASE_IMAGE}" >&2
docker build -f "${PROJECT_DIR}/containers/Dockerfile.experiments" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
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
