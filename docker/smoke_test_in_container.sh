#!/usr/bin/env bash
set -euo pipefail

cd /workspace

mkdir -p models data/calibration

if [[ ! -f models/resnet50.onnx ]]; then
  /workspace/scripts/download_resnet50_onnx.sh models/resnet50.onnx || true
fi

if [[ ! -f data/calibration/input_list.txt ]]; then
  printf 'sample_input.raw\n' > data/calibration/input_list.txt
fi

MODE="${PIPELINE_MODE:-mock}"
if [[ "$MODE" == "real" ]]; then
  if [[ ! -x "${QNN_SDK_ROOT:-/opt/qairt}/bin/qnn-onnx-converter" ]]; then
    echo "QNN SDK not found at ${QNN_SDK_ROOT:-/opt/qairt}. Mount SDK or set QAIRT_SDK_ARCHIVE."
    exit 1
  fi
  if [[ -z "${QAI_HUB_API_KEY:-}" ]]; then
    echo "QAI_HUB_API_KEY is required for real AI Hub profiling mode."
    exit 1
  fi
  edge-qnn-pipeline run --config configs/example_s24_real.yaml --log-level INFO
else
  edge-qnn-pipeline run --config configs/example_s24_run.yaml --log-level INFO
fi
