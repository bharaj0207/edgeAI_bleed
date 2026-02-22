# Low-Level Process Guide

## 1) Model Ingest

### ONNX input
- Source path copied to run workspace: `runs/<model>_<timestamp>/model/<name>.onnx`.

### PyTorch input
- Uses configurable `pytorch_export_command`.
- Template vars:
  - `{input_path}`
  - `{output_onnx}`
  - `{opset}`
  - `{input_shape}`
  - `{model_name}`

Example:

```bash
python scripts/export_pytorch.py --input {input_path} --output {output_onnx} --opset {opset} --input-shape {input_shape}
```

## 2) QNN Conversion

### 2.1 ONNX -> C++ + BIN

Tool: `qnn-onnx-converter`

Key flags assembled:
- `--input_network <model.onnx>`
- `--output_path <model.cpp>`
- `--binary_file <model.bin>`
- quantization knobs (when enabled):
  - `--weight_bw`
  - `--act_bw`
  - `--input_list`
  - `--use_per_channel_quantization`
  - `--bias_correction`
  - `--percentile_calibration_value`

### 2.2 Model library generation

Tool: `qnn-model-lib-generator`

Key flags assembled:
- `--cpp <model.cpp>`
- `--bin <model.bin>`
- `--lib_targets <toolchain>`
- `--output_dir <model_lib/>`

### 2.3 Context binary generation

Tool: `qnn-context-binary-generator`

Key flags assembled:
- `--backend <libQnnHtp.so or libQnnCpu.so>`
- `--model <model_lib/>`
- `--binary_file <*.serialized.bin>`
- optional: `--config_file <backend_extension.json>`
- optional target-specific: `--soc_id`, `--htp_arch`

## 3) Profiling

Two modes:
- `mock`: deterministic synthetic latency for dry-loop validation.
- `qai_hub_sdk` (default real mode): use official Qualcomm SDK.
- `qai_hub_rest` (fallback): direct REST integration for custom endpoint deployments.

Expected flow for `qai_hub_sdk` mode:
1. `qai_hub.upload_model(<local_context_binary>)`
2. `qai_hub.submit_profile_job(...)`
3. `ProfileJob.wait(...)`
4. `ProfileJob.download_profile()`
5. parse latency/throughput/power/memory from returned profile payload

Expected flow for `qai_hub_rest` mode:
1. `POST /v1/artifacts` with context binary.
2. `POST /v1/jobs` with target + artifact metadata.
3. Poll `GET /v1/jobs/<id>` until success/fail.
4. Parse inline `metrics` or fetch `GET /v1/jobs/<id>/metrics`.

## 4) Optimization Loop Policy

Current policy is deterministic and easy to replace.

Action progression:
1. baseline int8 per-channel quantization
2. percentile calibration (`99.9`)
3. mixed precision (`int4` weights)
4. input resize (0.9x)
5. architecture hook command
6. input resize (0.8x)

Stop conditions:
- latency <= target
- max iterations reached
- no further actions available

## 5) Iteration Analysis Agent

After each profile result, the pipeline runs an analyzer node:
- `rule_based` mode (default): deterministic bottleneck heuristics
- `openai` mode (optional): LangChain + OpenAI JSON analysis

Output per iteration:
- `analysis.json`

This analysis is persisted and can be used by future planner agents.

## 6) Artifacts Per Iteration

Path: `runs/<model>_<timestamp>/iter_<NN>/`

Includes:
- `action.json`
- `profile_metrics.json`
- `conversion/`
  - `<model>.cpp`
  - `<model>.bin`
  - `model_lib/`
  - `<context>.serialized.bin`
  - `converter.log`
  - `context_generator.log`

Top-level run summary:
- `run_metadata.json`
- `summary.json`
- `final_report.md`

## 7) Target-Specific Guidance (Galaxy S24 Family)

Baseline target preset:
- SoC: `SM8650`
- backend: `HTP`
- `htp_soc_id`: `43`
- `htp_arch`: `v75`

If your QAIRT/QNN build expects different SoC IDs/arch labels, override these values in config.

## 8) Integrating Additional Agents in LangGraph

Recommended insertion points:
- After `profile_model`: add report-analysis agent node that classifies bottleneck patterns.
- Replace `select_action`: add planner agent that emits structured `OptimizationAction`.
- Before `convert_model`: add policy guardrail node to enforce constraints (accuracy floor, power cap).

Keep agent I/O structured and stateful so execution remains deterministic and auditable.
