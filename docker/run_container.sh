#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-edge-qnn-pipeline:latest}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not in PATH"
  exit 1
fi

docker build -f "$PROJECT_ROOT/docker/Dockerfile" -t "$IMAGE_NAME" "$PROJECT_ROOT"

docker run --rm -it \
  -v "$PROJECT_ROOT":/workspace \
  -e QNN_SDK_ROOT="${QNN_SDK_ROOT:-/opt/qairt}" \
  -e QAIRT_SDK_ARCHIVE="${QAIRT_SDK_ARCHIVE:-}" \
  -e QAI_HUB_API_KEY="${QAI_HUB_API_KEY:-}" \
  -e QAI_HUB_BASE_URL="${QAI_HUB_BASE_URL:-}" \
  "$IMAGE_NAME" \
  /workspace/docker/smoke_test_in_container.sh
