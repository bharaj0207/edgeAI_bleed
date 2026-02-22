# Edge QNN Runtime Optimization Pipeline

A modular, config-driven framework to:
- ingest PyTorch or ONNX models,
- convert them into Qualcomm QNN/QAIRT artifacts and context binaries,
- profile runtime on device targets (mock backend included, QAI Hub backend scaffolded),
- iterate optimization decisions with a LangGraph control loop until latency targets are met.

Initial target profile included:
- `Samsung Galaxy S24 (Family)` / Snapdragon 8 Gen 3 (`SM8650`)

The architecture is built for extension to more SoCs, profiling backends, and optimization agents.

## What This Solves

You asked for a PC-first pipeline where conversion happens locally (QNN SDK), then profiling can run on AI Hub targets, with iterative optimization decisions made by agents. This repository provides:
- strict stage separation,
- pluggable target profiles,
- pluggable optimization strategy,
- LangGraph orchestration to support additional AI agents in the loop.

## Repository Layout

- `src/edge_qnn_pipeline/config.py`: typed config contracts
- `src/edge_qnn_pipeline/ingest/model_loader.py`: PyTorch/ONNX ingest and export hooks
- `src/edge_qnn_pipeline/conversion/qnn_backend.py`: QNN converter/lib/context binary generation
- `src/edge_qnn_pipeline/profiling/aihub_client.py`: profiling backend interface (`mock` + `qai_hub_sdk` + `qai_hub_rest`)
- `src/edge_qnn_pipeline/optimization/strategy.py`: iterative optimization policy engine
- `src/edge_qnn_pipeline/optimization/iteration_analyzer.py`: agent-style analysis of each iteration
- `src/edge_qnn_pipeline/workflow/graph.py`: LangGraph state machine
- `src/edge_qnn_pipeline/reports/writer.py`: per-iteration and final reports
- `configs/example_s24_run.yaml`: full pipeline config example
- `configs/example_s24_real.yaml`: real conversion + AI Hub profiling config
- `scripts/export_pytorch.py`: PyTorch -> ONNX helper
- `scripts/architecture_tune.py`: architecture-tuning hook placeholder
- `scripts/download_resnet50_onnx.sh`: fetches a sample ResNet-50 ONNX
- `docker/Dockerfile`: reproducible container environment
- `docker/smoke_test_in_container.sh`: container smoke-test runner

## Prerequisites

1. Python 3.10+
2. Qualcomm QNN/QAIRT SDK installed locally
3. `QNN_SDK_ROOT` exported to SDK root path
4. Optional: API key env var for AI Hub (`QAI_HUB_API_KEY` by default)

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional LLM analyzer support:

```bash
pip install -e ".[agent]"
```

## Run

List target profiles:

```bash
edge-qnn-pipeline list-targets
```

Run optimization loop:

```bash
edge-qnn-pipeline run --config configs/example_s24_run.yaml
```

By default the example config uses `dry_run: true` and `aihub.mode: mock` so you can validate the pipeline without cloud/device calls.

## Docker Environment

Build image:

```bash
docker build -f docker/Dockerfile -t edge-qnn-pipeline:latest .
```

Run container smoke test (mock mode):

```bash
docker run --rm -it -v "$PWD":/workspace edge-qnn-pipeline:latest /workspace/docker/smoke_test_in_container.sh
```

Or use helper:

```bash
./docker/run_container.sh
```

For real SDK + AI Hub runs, see:
- `docker/README.md`
- `configs/example_s24_real.yaml`
- `.env.example`

## Switch To Real AI Hub

In config:
- use `configs/example_s24_real.yaml` (already set to real mode using `qai_hub` SDK)
- set `aihub.project` and `aihub.device`

In environment:

```bash
export QAI_HUB_API_KEY=<your-key>
export QNN_SDK_ROOT=<path-to-qnn-sdk>
```

Install AI Hub SDK extras if needed:

```bash
pip install -e ".[qaihub]"
```

## How Iterative Optimization Works

Each iteration executes:
1. choose optimization action (quantization/shape/architecture hook),
2. convert ONNX -> QNN artifacts -> context binary,
3. profile latency,
4. analyze performance report (rule-based or LLM),
5. compare against latency target,
6. continue or stop.

Current policy sequence:
- baseline int8 per-channel
- percentile calibration
- mixed precision int4 weights
- input resize (if enabled)
- architecture tuning hook (if configured)
- additional resize step

You can replace this with a learned agent policy later.

## Extending For New Hardware

1. Add a new profile in `src/edge_qnn_pipeline/targets/registry.py`
2. Add target-specific backend extension JSON under `configs/backend_extensions/`
3. Use target name in YAML config

## LangGraph Agent Expansion

Recommended next agents to add:
- report analyst agent (reads iteration metrics and identifies bottlenecks)
- optimization planner agent (chooses next action from candidate set)
- architecture search agent (produces pruning/distillation proposals)

The workflow is structured so these can be inserted as new nodes with minimal refactoring.

## Notes

- Default AI Hub integration uses official `qai_hub` SDK (`submit_profile_job`, `download_profile`).
- A REST fallback mode (`aihub.mode: qai_hub_rest`) is kept for custom deployments that require direct HTTP APIs.
- `scripts/architecture_tune.py` is intentionally a hook and does not modify model topology by itself.
