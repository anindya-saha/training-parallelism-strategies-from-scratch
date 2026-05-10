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
# Load your env.dev (REGISTRY, etc.) before launching this script so the
# variables below pick up your overrides.

REGISTRY="${REGISTRY:?Set REGISTRY in your env.dev}"
PUSH=true
TARGET="runtime"

VERSIONS_ENV="${PROJECT_DIR}/containers/docker/pytorch/versions-cuda.env"

# -- Usage --------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Build and optionally push the DLC-style PyTorch base image (CUDA).

This builds containers/docker/pytorch/Dockerfile.cuda with build context
set to containers/ so all COPY paths resolve correctly.

Options:
  --registry URL            Container registry
  --base-repo REPO          Image repository (default: <registry>/<user>/dist-train/base)
  --target STAGE            Docker target stage (default: runtime)
  --versions-env FILE       Path to versions-cuda.env (default: containers/docker/pytorch/versions-cuda.env)
  --no-push                 Build only, do not push to registry
  -h, --help                Show this help

Outputs (written to stdout on the last line):
  BASE_IMAGE_URI=<image-uri>

Examples:
  # Build and push
  $0

  # Build only, no push
  $0 --no-push

  # Use as input to build_image.sh via eval
  eval \$($0)
EOF
  exit 1
}

# -- Parse args ---------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case $1 in
    --registry)        REGISTRY="$2"; shift 2 ;;
    --base-repo)       BASE_REPO="$2"; shift 2 ;;
    --target)          TARGET="$2"; shift 2 ;;
    --versions-env)    VERSIONS_ENV="$2"; shift 2 ;;
    --no-push)         PUSH=false; shift ;;
    -h|--help)         usage ;;
    *)                 echo "Unknown option: $1"; usage ;;
  esac
done

BASE_REPO="${BASE_REPO:-${REGISTRY}/${CURRENT_USER}/dist-train/base}"

# -- Source versions -----------------------------------------------------------

if [[ ! -f "${VERSIONS_ENV}" ]]; then
  echo "Error: versions file not found: ${VERSIONS_ENV}" >&2
  exit 1
fi

echo "==> Sourcing versions from ${VERSIONS_ENV}" >&2
# shellcheck disable=SC1090
source "${VERSIONS_ENV}"

# -- Build image --------------------------------------------------------------

BUILD_CONTEXT="${PROJECT_DIR}/containers"
DOCKERFILE="${PROJECT_DIR}/containers/docker/pytorch/Dockerfile.cuda"

echo "==> Building base image (target: ${TARGET})..." >&2
echo "    Dockerfile: ${DOCKERFILE}" >&2
echo "    Context:    ${BUILD_CONTEXT}" >&2
echo "    CUDA=${CUDA_VERSION} PyTorch=${TORCH_VERSION} Python=${PYTHON_VERSION}" >&2

docker build -f "${DOCKERFILE}" \
  --target "${TARGET}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --build-arg TORCH_VERSION="${TORCH_VERSION}" \
  --build-arg TRANSFORMER_ENGINE_VERSION="${TRANSFORMER_ENGINE_VERSION}" \
  --build-arg GDRCOPY_VERSION="${GDRCOPY_VERSION}" \
  --build-arg EFA_VERSION="${EFA_VERSION}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  --build-arg DLC_MAJOR_VERSION="${DLC_MAJOR_VERSION}" \
  --build-arg DLC_MINOR_VERSION="${DLC_MINOR_VERSION}" \
  -t "${BASE_REPO}:temp" \
  "${BUILD_CONTEXT}" >&2

IMAGE_SHA=$(docker inspect --format='{{.Id}}' "${BASE_REPO}:temp")
IMAGE_SHA="${IMAGE_SHA#sha256:}"
BASE_IMAGE_URI="${BASE_REPO}:${IMAGE_SHA}"
docker tag "${BASE_REPO}:temp" "${BASE_IMAGE_URI}" >&2
docker rmi "${BASE_REPO}:temp" >/dev/null 2>&1 || true

# -- Push ----------------------------------------------------------------------

if [[ "${PUSH}" == "true" ]]; then
  echo "==> Pushing image..." >&2
  docker push "${BASE_IMAGE_URI}" >&2
fi

# -- Output image URI (machine-readable) --------------------------------------

echo "==> Base image: ${BASE_IMAGE_URI}" >&2
echo "BASE_IMAGE_URI=${BASE_IMAGE_URI}"
