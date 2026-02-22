from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    framework: str
    model_path: Path
    pytorch_export_command: str | None = None
    onnx_opset: int = 17
    input_shape: list[int] = field(default_factory=list)
    dynamic_input: bool = False
    model_name: str = "model"


@dataclass
class TargetConfig:
    name: str
    soc: str
    device_family: str
    qnn_backend: str = "htp"
    htp_soc_id: int | None = None
    htp_arch: str | None = None
    context_binary_name: str = "model.serialized.bin"
    backend_extensions_json: Path | None = None


@dataclass
class QuantizationConfig:
    enabled: bool = True
    scheme: str = "int8_per_channel"
    calibration_dataset: Path | None = None
    use_per_channel: bool = True
    use_bias_correction: bool = True
    percentile: float | None = None


@dataclass
class QNNConfig:
    sdk_root: Path
    toolchain: str = "aarch64-android"
    output_dir: Path = Path("artifacts")
    converter_tool: str = "qnn-onnx-converter"
    model_lib_generator_tool: str = "qnn-model-lib-generator"
    context_binary_generator_tool: str = "qnn-context-binary-generator"
    net_run_tool: str = "qnn-net-run"
    extra_converter_args: list[str] = field(default_factory=list)
    extra_context_args: list[str] = field(default_factory=list)


@dataclass
class AIHubConfig:
    mode: str = "mock"
    base_url: str | None = None
    api_key_env: str = "QAI_HUB_API_KEY"
    project: str | None = None
    polling_interval_sec: int = 10
    timeout_sec: int = 1800


@dataclass
class OptimizationConfig:
    target_latency_ms: float
    max_iterations: int = 8
    fail_fast: bool = False
    allow_architecture_tuning: bool = True
    architecture_tuning_command: str | None = None
    allow_input_resize: bool = True
    min_input_edge: int = 128


@dataclass
class AgentConfig:
    enabled: bool = False
    mode: str = "rule_based"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 400


@dataclass
class PipelineConfig:
    model: ModelConfig
    target: TargetConfig
    quantization: QuantizationConfig
    qnn: QNNConfig
    aihub: AIHubConfig
    optimization: OptimizationConfig
    agent: AgentConfig
    run_dir: Path = Path("runs")
    dry_run: bool = True


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _to_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def load_config(path: str | Path) -> PipelineConfig:
    data = yaml.safe_load(Path(path).read_text())
    data = _expand_env(data)

    model = data["model"]
    target = data["target"]
    quantization = data.get("quantization", {})
    qnn = data["qnn"]
    aihub = data.get("aihub", {})
    optimization = data["optimization"]
    agent = data.get("agent", {})

    return PipelineConfig(
        model=ModelConfig(
            framework=model["framework"],
            model_path=Path(model["model_path"]),
            pytorch_export_command=model.get("pytorch_export_command"),
            onnx_opset=model.get("onnx_opset", 17),
            input_shape=model.get("input_shape", []),
            dynamic_input=model.get("dynamic_input", False),
            model_name=model.get("model_name", "model"),
        ),
        target=TargetConfig(
            name=target["name"],
            soc=target["soc"],
            device_family=target["device_family"],
            qnn_backend=target.get("qnn_backend", "htp"),
            htp_soc_id=target.get("htp_soc_id"),
            htp_arch=target.get("htp_arch"),
            context_binary_name=target.get("context_binary_name", "model.serialized.bin"),
            backend_extensions_json=_to_path(target.get("backend_extensions_json")),
        ),
        quantization=QuantizationConfig(
            enabled=quantization.get("enabled", True),
            scheme=quantization.get("scheme", "int8_per_channel"),
            calibration_dataset=_to_path(quantization.get("calibration_dataset")),
            use_per_channel=quantization.get("use_per_channel", True),
            use_bias_correction=quantization.get("use_bias_correction", True),
            percentile=quantization.get("percentile"),
        ),
        qnn=QNNConfig(
            sdk_root=Path(qnn["sdk_root"]),
            toolchain=qnn.get("toolchain", "aarch64-android"),
            output_dir=Path(qnn.get("output_dir", "artifacts")),
            converter_tool=qnn.get("converter_tool", "qnn-onnx-converter"),
            model_lib_generator_tool=qnn.get("model_lib_generator_tool", "qnn-model-lib-generator"),
            context_binary_generator_tool=qnn.get("context_binary_generator_tool", "qnn-context-binary-generator"),
            net_run_tool=qnn.get("net_run_tool", "qnn-net-run"),
            extra_converter_args=qnn.get("extra_converter_args", []),
            extra_context_args=qnn.get("extra_context_args", []),
        ),
        aihub=AIHubConfig(
            mode=aihub.get("mode", "mock"),
            base_url=aihub.get("base_url"),
            api_key_env=aihub.get("api_key_env", "QAI_HUB_API_KEY"),
            project=aihub.get("project"),
            polling_interval_sec=aihub.get("polling_interval_sec", 10),
            timeout_sec=aihub.get("timeout_sec", 1800),
        ),
        optimization=OptimizationConfig(
            target_latency_ms=optimization["target_latency_ms"],
            max_iterations=optimization.get("max_iterations", 8),
            fail_fast=optimization.get("fail_fast", False),
            allow_architecture_tuning=optimization.get("allow_architecture_tuning", True),
            architecture_tuning_command=optimization.get("architecture_tuning_command"),
            allow_input_resize=optimization.get("allow_input_resize", True),
            min_input_edge=optimization.get("min_input_edge", 128),
        ),
        agent=AgentConfig(
            enabled=agent.get("enabled", False),
            mode=agent.get("mode", "rule_based"),
            model=agent.get("model"),
            temperature=agent.get("temperature", 0.0),
            max_tokens=agent.get("max_tokens", 400),
        ),
        run_dir=Path(data.get("run_dir", "runs")),
        dry_run=data.get("dry_run", True),
    )
