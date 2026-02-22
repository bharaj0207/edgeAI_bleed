# Docker Environment

## What it provides

- Python 3.11 runtime
- system-level build/runtime dependencies
- editable install of this project (`edge-qnn-pipeline`)
- optional QAIRT/QNN SDK install from a mounted archive

## Build

```bash
docker build -f docker/Dockerfile -t edge-qnn-pipeline:latest .
```

## Mock smoke test (no SDK/API required)

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  edge-qnn-pipeline:latest \
  /workspace/docker/smoke_test_in_container.sh
```

## Real run (SDK + AI Hub)

```bash
export QAI_HUB_API_KEY=<your-key>

# Put your SDK archive in the repo root or mount it from elsewhere
export QAIRT_SDK_ARCHIVE=/workspace/qairt-sdk.tar.gz

docker run --rm -it \
  -v "$PWD":/workspace \
  -e QAI_HUB_API_KEY \
  -e QAIRT_SDK_ARCHIVE \
  edge-qnn-pipeline:latest \
  bash -lc 'PIPELINE_MODE=real /workspace/docker/smoke_test_in_container.sh'
```

If SDK is already mounted/extracted at `/opt/qairt` inside container, set:

```bash
export QNN_SDK_ROOT=/opt/qairt
```

## Notes

- `docker/install_qairt_sdk.sh` supports `.tar.gz`, `.tgz`, and `.zip` archives.
- Real profiling uses `configs/example_s24_real.yaml`.
- Mock smoke test uses `configs/example_s24_run.yaml`.
