from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversionArtifacts:
    onnx_model: Path
    cpp_model: Path
    bin_model: Path
    model_lib_dir: Path
    context_binary: Path
    converter_log: Path
    context_log: Path
