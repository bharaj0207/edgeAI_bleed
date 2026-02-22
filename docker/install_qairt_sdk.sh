#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <qairt-sdk-archive.(tar.gz|tgz|zip)>"
  exit 1
fi

ARCHIVE="$1"
DEST_ROOT="${QNN_SDK_ROOT:-/opt/qairt}"
TMP_DIR="/tmp/qairt_extract"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "QAIRT archive not found: $ARCHIVE"
  exit 1
fi

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
mkdir -p "$DEST_ROOT"

case "$ARCHIVE" in
  *.tar.gz|*.tgz)
    tar -xzf "$ARCHIVE" -C "$TMP_DIR"
    ;;
  *.zip)
    unzip -q "$ARCHIVE" -d "$TMP_DIR"
    ;;
  *)
    echo "Unsupported archive format: $ARCHIVE"
    exit 1
    ;;
esac

SDK_DIR=""
if [[ -d "$TMP_DIR/bin" ]]; then
  SDK_DIR="$TMP_DIR"
else
  SDK_DIR="$(find "$TMP_DIR" -maxdepth 3 -type d -name bin -print | sed 's#/bin$##' | head -n 1)"
fi

if [[ -z "$SDK_DIR" ]]; then
  echo "Could not locate SDK root with bin/ directory in archive."
  exit 1
fi

rm -rf "$DEST_ROOT"/*
cp -R "$SDK_DIR"/* "$DEST_ROOT"/

if [[ ! -x "$DEST_ROOT/bin/qnn-onnx-converter" ]]; then
  echo "SDK installed but qnn-onnx-converter not found under $DEST_ROOT/bin"
  exit 1
fi

echo "QAIRT/QNN SDK installed to $DEST_ROOT"
