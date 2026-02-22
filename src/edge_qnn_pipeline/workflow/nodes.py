from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from edge_qnn_pipeline.command import CommandRunner, dump_json
from edge_qnn_pipeline.config import PipelineConfig
from edge_qnn_pipeline.conversion.qnn_backend import QNNConverter
from edge_qnn_pipeline.ingest.model_loader import ModelIngestor
from edge_qnn_pipeline.optimization.iteration_analyzer import IterationAnalyzer
from edge_qnn_pipeline.optimization.strategy import OptimizationPlanner
from edge_qnn_pipeline.profiling.aihub_client import ProfilerBackend
from edge_qnn_pipeline.reports.writer import write_final_report, write_iteration_metrics
from edge_qnn_pipeline.workflow.state import PipelineState


LOG = logging.getLogger(__name__)


class PipelineNodes:
    def __init__(
        self,
        config: PipelineConfig,
        runner: CommandRunner,
        profiler: ProfilerBackend,
        planner: OptimizationPlanner,
        analyzer: IterationAnalyzer,
    ):
        self.config = config
        self.runner = runner
        self.profiler = profiler
        self.planner = planner
        self.analyzer = analyzer
        self.ingestor = ModelIngestor(runner)
        self.converter = QNNConverter(config.qnn, runner)

    def initialize(self, _: PipelineState) -> PipelineState:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = self.config.run_dir / f"{self.config.model.model_name}_{run_id}"
        run_root.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model": str(self.config.model.model_path),
            "framework": self.config.model.framework,
            "target": self.config.target.name,
            "soc": self.config.target.soc,
            "dry_run": self.config.dry_run,
        }
        dump_json(run_root / "run_metadata.json", metadata)

        return {
            "iteration": 0,
            "run_root": run_root,
            "history": [],
            "analysis_history": [],
            "action_history": [],
            "continue_loop": True,
            "stop_reason": "",
        }

    def prepare_model(self, state: PipelineState) -> PipelineState:
        if "onnx_model" in state:
            return {}
        run_root = state["run_root"]
        onnx = self.ingestor.prepare_onnx(self.config.model, run_root / "model")
        return {"onnx_model": onnx}

    def select_action(self, state: PipelineState) -> PipelineState:
        action = self.planner.propose_action(state["history"], state["iteration"])
        if action is None:
            return {
                "current_action": None,
                "continue_loop": False,
                "stop_reason": "No additional optimization action available.",
            }
        return {"current_action": action}

    def apply_action(self, state: PipelineState) -> PipelineState:
        action = state.get("current_action")
        if action is None:
            return {}

        run_root = state["run_root"]
        iter_dir = run_root / f"iter_{state['iteration']:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        applied = self.planner.apply_action(action, iter_dir)
        action_entry = {
            "iteration": state["iteration"],
            "action": action.name,
            "reason": action.reason,
            "applied": applied,
        }
        action_path = iter_dir / "action.json"
        dump_json(action_path, action_entry)

        return {
            "iteration_dir": iter_dir,
            "action_history": state["action_history"] + [action_entry],
        }

    def convert_model(self, state: PipelineState) -> PipelineState:
        iter_dir = state["iteration_dir"]
        artifacts = self.converter.convert(
            onnx_model=state["onnx_model"],
            quantization=self.config.quantization,
            target=self.config.target,
            output_dir=iter_dir / "conversion",
            model_name=self.config.model.model_name,
        )
        return {"conversion_artifacts": artifacts}

    def profile_model(self, state: PipelineState) -> PipelineState:
        artifacts = state["conversion_artifacts"]
        profile = self.profiler.profile(
            context_binary=artifacts.context_binary,
            target=self.config.target,
            iteration=state["iteration"],
        )

        write_iteration_metrics(
            path=state["iteration_dir"] / "profile_metrics.json",
            iteration=state["iteration"],
            profile=profile,
            action=state["action_history"][-1] if state["action_history"] else None,
        )
        return {"latest_profile": profile}

    def analyze_profile(self, state: PipelineState) -> PipelineState:
        history_with_latest = state["history"] + [state["latest_profile"]]
        last_action = "none"
        if state["action_history"]:
            last_action = state["action_history"][-1]["action"]

        analysis = self.analyzer.analyze(
            history=history_with_latest,
            target_latency_ms=self.config.optimization.target_latency_ms,
            last_action_name=last_action,
        )
        dump_json(state["iteration_dir"] / "analysis.json", asdict(analysis))
        return {
            "latest_analysis": analysis,
            "analysis_history": state["analysis_history"] + [asdict(analysis)],
        }

    def evaluate(self, state: PipelineState) -> PipelineState:
        history = state["history"] + [state["latest_profile"]]
        should_stop, reason = self.planner.should_stop(state["latest_profile"], state["iteration"])

        next_state: PipelineState = {
            "history": history,
            "continue_loop": not should_stop,
            "stop_reason": reason,
        }
        if not should_stop:
            next_state["iteration"] = state["iteration"] + 1
        return next_state

    def finalize(self, state: PipelineState) -> PipelineState:
        run_root = state["run_root"]
        write_final_report(
            path=run_root / "final_report.md",
            history=state["history"],
            target_latency_ms=self.config.optimization.target_latency_ms,
            stop_reason=state["stop_reason"],
        )
        dump_json(
            run_root / "summary.json",
            {
                "iterations": len(state["history"]),
                "best_latency_ms": min(item.latency_ms for item in state["history"]),
                "target_latency_ms": self.config.optimization.target_latency_ms,
                "stop_reason": state["stop_reason"],
                "analysis_history": state.get("analysis_history", []),
            },
        )
        LOG.info("Run complete. Output written to %s", run_root)
        return {}
