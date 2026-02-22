from __future__ import annotations

import logging
import shlex
import shutil
from pathlib import Path

from edge_qnn_pipeline.command import CommandRunner
from edge_qnn_pipeline.config import ModelConfig


LOG = logging.getLogger(__name__)


class ModelIngestor:
    def __init__(self, runner: CommandRunner):
        self.runner = runner

    def prepare_onnx(self, model: ModelConfig, work_dir: Path) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        output_onnx = work_dir / f"{model.model_name}.onnx"

        if model.framework.lower() == "onnx":
            LOG.info("Using ONNX model directly: %s", model.model_path)
            if not self.runner.dry_run:
                shutil.copy2(model.model_path, output_onnx)
            return output_onnx

        if model.framework.lower() == "pytorch":
            if not model.pytorch_export_command:
                raise ValueError(
                    "model.pytorch_export_command is required when framework is 'pytorch'."
                )
            rendered = model.pytorch_export_command.format(
                input_path=str(model.model_path),
                output_onnx=str(output_onnx),
                opset=model.onnx_opset,
                input_shape=",".join(str(v) for v in model.input_shape),
                model_name=model.model_name,
            )
            LOG.info("Exporting PyTorch model with command template")
            self.runner.run(shlex.split(rendered))
            return output_onnx

        raise ValueError("model.framework must be one of: onnx, pytorch")
