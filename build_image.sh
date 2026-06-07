#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${1:-mcvrp-pdtw-v6}
IMAGE_TAG=${2:-latest}
TAR_NAME=${3:-${IMAGE_NAME}_${IMAGE_TAG}.tar}
PLATFORM=${PLATFORM:-linux/amd64}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if docker buildx version >/dev/null 2>&1; then
  docker buildx build \
    --platform "${PLATFORM}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    --load \
    "${SCRIPT_DIR}"
else
  docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "${SCRIPT_DIR}"
fi

docker save -o "${TAR_NAME}" "${IMAGE_NAME}:${IMAGE_TAG}"

echo "Built image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Saved tar : ${TAR_NAME}"
echo "Platform  : ${PLATFORM}"
