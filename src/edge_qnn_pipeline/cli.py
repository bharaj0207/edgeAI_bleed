from __future__ import annotations

import argparse

from edge_qnn_pipeline.command import CommandRunner
from edge_qnn_pipeline.config import load_config
from edge_qnn_pipeline.logging_utils import configure_logging
from edge_qnn_pipeline.optimization.iteration_analyzer import build_iteration_analyzer
from edge_qnn_pipeline.optimization.strategy import OptimizationPlanner
from edge_qnn_pipeline.profiling.aihub_client import build_profiler
from edge_qnn_pipeline.targets.registry import hydrate_target_config, list_targets
from edge_qnn_pipeline.workflow.graph import PipelineGraph
from edge_qnn_pipeline.workflow.nodes import PipelineNodes


def _cmd_list_targets() -> int:
    for target in list_targets():
        print(f"{target.name}: {target.device_family} ({target.soc}) [{target.qnn_backend}]")
    return 0


def _cmd_run(config_path: str, log_level: str) -> int:
    configure_logging(log_level)
    config = load_config(config_path)
    config.target = hydrate_target_config(config.target)
    if not config.dry_run and "$" in str(config.qnn.sdk_root):
        raise ValueError(
            "qnn.sdk_root contains unresolved environment variables. "
            "Export QNN_SDK_ROOT (or update config) before running with dry_run=false."
        )

    runner = CommandRunner(dry_run=config.dry_run)
    profiler = build_profiler(config.aihub)
    planner = OptimizationPlanner(config.optimization, config.model, config.quantization, runner)
    analyzer = build_iteration_analyzer(config.agent, config.optimization)
    nodes = PipelineNodes(
        config=config,
        runner=runner,
        profiler=profiler,
        planner=planner,
        analyzer=analyzer,
    )
    graph = PipelineGraph(nodes).compile()

    final_state = graph.invoke({})
    run_root = final_state["run_root"]
    best = min(item.latency_ms for item in final_state["history"])
    print(f"Run completed: {run_root}")
    print(f"Best latency: {best:.3f} ms")
    print(f"Stop reason: {final_state['stop_reason']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-qnn-pipeline",
        description="LangGraph-driven QNN conversion and profiling pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Execute optimization pipeline")
    run.add_argument("--config", required=True, help="Path to YAML config")
    run.add_argument("--log-level", default="INFO", help="Logging level")

    sub.add_parser("list-targets", help="List built-in target profiles")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-targets":
        return _cmd_list_targets()
    if args.command == "run":
        return _cmd_run(args.config, args.log_level)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
