from __future__ import annotations

import logging
from pathlib import Path

from edge_qnn_pipeline.command import CommandRunner
from edge_qnn_pipeline.config import QNNConfig, QuantizationConfig, TargetConfig
from edge_qnn_pipeline.conversion.artifacts import ConversionArtifacts


LOG = logging.getLogger(__name__)


class QNNConverter:
    def __init__(self, qnn: QNNConfig, runner: CommandRunner):
        self.qnn = qnn
        self.runner = runner

    def convert(
        self,
        onnx_model: Path,
        quantization: QuantizationConfig,
        target: TargetConfig,
        output_dir: Path,
        model_name: str,
    ) -> ConversionArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

        cpp_model = output_dir / f"{model_name}.cpp"
        bin_model = output_dir / f"{model_name}.bin"
        model_lib_dir = output_dir / "model_lib"
        context_binary = output_dir / target.context_binary_name
        converter_log = output_dir / "converter.log"
        context_log = output_dir / "context_generator.log"

        converter_cmd = self._build_converter_command(
            onnx_model=onnx_model,
            cpp_model=cpp_model,
            bin_model=bin_model,
            quantization=quantization,
        )
        converter_result = self.runner.run(converter_cmd)
        converter_log.write_text(converter_result.stdout + "\n" + converter_result.stderr)

        model_lib_cmd = self._build_model_lib_command(cpp_model=cpp_model, bin_model=bin_model, model_lib_dir=model_lib_dir)
        self.runner.run(model_lib_cmd)

        context_cmd = self._build_context_binary_command(
            model_lib_dir=model_lib_dir,
            context_binary=context_binary,
            target=target,
        )
        context_result = self.runner.run(context_cmd)
        context_log.write_text(context_result.stdout + "\n" + context_result.stderr)

        return ConversionArtifacts(
            onnx_model=onnx_model,
            cpp_model=cpp_model,
            bin_model=bin_model,
            model_lib_dir=model_lib_dir,
            context_binary=context_binary,
            converter_log=converter_log,
            context_log=context_log,
        )

    def _build_converter_command(
        self,
        onnx_model: Path,
        cpp_model: Path,
        bin_model: Path,
        quantization: QuantizationConfig,
    ) -> list[str]:
        converter = self._resolve_tool(self.qnn.converter_tool)
        command = [
            str(converter),
            "--input_network",
            str(onnx_model),
            "--output_path",
            str(cpp_model),
            "--binary_file",
            str(bin_model),
        ]

        if quantization.enabled:
            weight_bw = "8"
            act_bw = "8"
            if quantization.scheme == "int4_mixed_precision":
                weight_bw = "4"
            command.extend(["--weight_bw", weight_bw, "--act_bw", act_bw])
            if quantization.calibration_dataset:
                command.extend(["--input_list", str(quantization.calibration_dataset)])
            if quantization.use_per_channel:
                command.append("--use_per_channel_quantization")
            if quantization.use_bias_correction:
                command.append("--bias_correction")
            if quantization.percentile is not None:
                command.extend(["--percentile_calibration_value", str(quantization.percentile)])

        command.extend(self.qnn.extra_converter_args)
        return command

    def _build_model_lib_command(self, cpp_model: Path, bin_model: Path, model_lib_dir: Path) -> list[str]:
        generator = self._resolve_tool(self.qnn.model_lib_generator_tool)
        return [
            str(generator),
            "--cpp",
            str(cpp_model),
            "--bin",
            str(bin_model),
            "--lib_targets",
            self.qnn.toolchain,
            "--output_dir",
            str(model_lib_dir),
        ]

    def _build_context_binary_command(
        self,
        model_lib_dir: Path,
        context_binary: Path,
        target: TargetConfig,
    ) -> list[str]:
        generator = self._resolve_tool(self.qnn.context_binary_generator_tool)
        backend = self._resolve_backend_library(target.qnn_backend)

        command = [
            str(generator),
            "--backend",
            str(backend),
            "--model",
            str(model_lib_dir),
            "--binary_file",
            str(context_binary),
        ]

        if target.backend_extensions_json:
            command.extend(["--config_file", str(target.backend_extensions_json)])

        if target.htp_soc_id is not None:
            command.extend(["--soc_id", str(target.htp_soc_id)])
        if target.htp_arch:
            command.extend(["--htp_arch", str(target.htp_arch)])

        command.extend(self.qnn.extra_context_args)
        return command

    def _resolve_tool(self, tool_name: str) -> Path:
        return self.qnn.sdk_root / "bin" / tool_name

    def _resolve_backend_library(self, backend_name: str) -> Path:
        if backend_name.lower() == "htp":
            return self.qnn.sdk_root / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
        if backend_name.lower() == "cpu":
            return self.qnn.sdk_root / "lib" / "x86_64-linux-clang" / "libQnnCpu.so"
        raise ValueError(f"Unsupported backend '{backend_name}'")
