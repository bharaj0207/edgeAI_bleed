from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_qnn_pipeline.command import CommandRunner
from edge_qnn_pipeline.config import ModelConfig, OptimizationConfig, QuantizationConfig
from edge_qnn_pipeline.profiling.types import ProfileResult


LOG = logging.getLogger(__name__)


@dataclass
class OptimizationAction:
    name: str
    reason: str
    updates: dict[str, Any]


class OptimizationPlanner:
    def __init__(
        self,
        optimization: OptimizationConfig,
        model: ModelConfig,
        quantization: QuantizationConfig,
        runner: CommandRunner,
    ):
        self.optimization = optimization
        self.runner = runner
        self.initial_input_shape = list(model.input_shape)
        self.current_shape = list(model.input_shape)
        self.quantization = quantization

    def should_stop(self, profile: ProfileResult, iteration: int) -> tuple[bool, str]:
        if profile.latency_ms <= self.optimization.target_latency_ms:
            return True, (
                f"Latency target met ({profile.latency_ms:.3f} ms <= "
                f"{self.optimization.target_latency_ms:.3f} ms)."
            )
        if iteration >= self.optimization.max_iterations:
            return True, "Reached maximum iterations."
        return False, "Continue optimizing."

    def propose_action(self, history: list[ProfileResult], iteration: int) -> OptimizationAction | None:
        if not history:
            return OptimizationAction(
                name="baseline_int8",
                reason="Establish quantized baseline.",
                updates={
                    "quantization.scheme": "int8_per_channel",
                    "quantization.use_per_channel": True,
                    "quantization.percentile": None,
                },
            )

        step = iteration
        if step == 1:
            return OptimizationAction(
                name="calibration_percentile",
                reason="Reduce activation outliers during calibration.",
                updates={"quantization.percentile": 99.9},
            )
        if step == 2:
            return OptimizationAction(
                name="mixed_precision_int4",
                reason="Reduce weight bandwidth pressure.",
                updates={"quantization.scheme": "int4_mixed_precision"},
            )
        if step == 3 and self.optimization.allow_input_resize:
            return self._input_resize_action(scale=0.9)
        if step == 4 and self.optimization.allow_architecture_tuning:
            if self.optimization.architecture_tuning_command:
                return OptimizationAction(
                    name="architecture_tuning",
                    reason="Run architecture-aware simplification hook.",
                    updates={"architecture_tuning": True},
                )
        if step == 5 and self.optimization.allow_input_resize:
            return self._input_resize_action(scale=0.8)

        return None

    def apply_action(self, action: OptimizationAction, work_dir: Path) -> dict[str, Any]:
        applied: dict[str, Any] = {}
        for key, value in action.updates.items():
            if key.startswith("quantization."):
                attribute = key.split(".", maxsplit=1)[1]
                setattr(self.quantization, attribute, value)
                applied[key] = value
                continue
            if key == "model.input_shape":
                self.current_shape = list(value)
                applied[key] = list(value)
                continue
            if key == "architecture_tuning" and value:
                self._run_architecture_tuning(work_dir)
                applied[key] = True

        LOG.info("Applied action '%s': %s", action.name, applied)
        return applied

    def _input_resize_action(self, scale: float) -> OptimizationAction | None:
        if len(self.current_shape) < 4:
            return None

        h = self.current_shape[-2]
        w = self.current_shape[-1]
        next_h = max(self.optimization.min_input_edge, int(h * scale))
        next_w = max(self.optimization.min_input_edge, int(w * scale))
        if next_h == h and next_w == w:
            return None

        shape = list(self.current_shape)
        shape[-2] = next_h
        shape[-1] = next_w

        return OptimizationAction(
            name=f"input_resize_{scale:.2f}",
            reason=f"Lower spatial resolution to reduce MACs ({h}x{w} -> {next_h}x{next_w}).",
            updates={"model.input_shape": shape},
        )

    def _run_architecture_tuning(self, work_dir: Path) -> None:
        command = self.optimization.architecture_tuning_command
        if not command:
            return
        rendered = command.format(work_dir=str(work_dir), input_shape=",".join(str(v) for v in self.current_shape))
        self.runner.run(shlex.split(rendered))
