from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from edge_qnn_pipeline.conversion.artifacts import ConversionArtifacts
from edge_qnn_pipeline.optimization.iteration_analyzer import IterationAnalysis
from edge_qnn_pipeline.optimization.strategy import OptimizationAction
from edge_qnn_pipeline.profiling.types import ProfileResult


class PipelineState(TypedDict, total=False):
    iteration: int
    run_root: Path
    iteration_dir: Path
    onnx_model: Path
    conversion_artifacts: ConversionArtifacts
    latest_profile: ProfileResult
    latest_analysis: IterationAnalysis
    history: list[ProfileResult]
    analysis_history: list[dict]
    action_history: list[dict]
    current_action: OptimizationAction | None
    stop_reason: str
    continue_loop: bool
