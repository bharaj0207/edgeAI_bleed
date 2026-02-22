#!/usr/bin/env bash
set -euo pipefail

OUT_PATH="${1:-models/resnet50.onnx}"
mkdir -p "$(dirname "$OUT_PATH")"

URLS=(
  "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
  "https://huggingface.co/onnxmodelzoo/resnet50-v1-7/resolve/main/resnet50-v1-7.onnx"
)

for url in "${URLS[@]}"; do
  echo "Attempting download: $url"
  if curl -fL "$url" -o "$OUT_PATH"; then
    if [[ -s "$OUT_PATH" ]]; then
      echo "Downloaded model to $OUT_PATH"
      exit 0
    fi
  fi
done

echo "Failed to download ResNet-50 ONNX model from all known URLs"
exit 1
