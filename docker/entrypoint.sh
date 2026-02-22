#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${QAIRT_SDK_ARCHIVE:-}" ]]; then
  /workspace/docker/install_qairt_sdk.sh "$QAIRT_SDK_ARCHIVE"
fi

if [[ -d "${QNN_SDK_ROOT:-/opt/qairt}/bin" ]]; then
  echo "[entrypoint] QNN SDK detected at ${QNN_SDK_ROOT:-/opt/qairt}"
else
  echo "[entrypoint] QNN SDK not found. Mount/install SDK for real conversion runs."
fi

exec "$@"
