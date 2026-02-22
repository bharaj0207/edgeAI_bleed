from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from edge_qnn_pipeline.command import dump_json
from edge_qnn_pipeline.optimization.analyzer import summarize_trend
from edge_qnn_pipeline.profiling.types import ProfileResult


def write_iteration_metrics(path: Path, iteration: int, profile: ProfileResult, action: dict | None = None) -> None:
    payload = {
        "iteration": iteration,
        "latency_ms": profile.latency_ms,
        "throughput_fps": profile.throughput_fps,
        "power_watts": profile.power_watts,
        "memory_mb": profile.memory_mb,
        "action": action or {},
        "raw": profile.raw,
    }
    dump_json(path, payload)


def write_final_report(path: Path, history: list[ProfileResult], target_latency_ms: float, stop_reason: str) -> None:
    trend = summarize_trend(history)
    now = datetime.now(timezone.utc).isoformat()

    lines = [
        f"# Edge QNN Optimization Report ({now})",
        "",
        f"- Iterations run: {len(history)}",
        f"- Target latency: {target_latency_ms:.3f} ms",
        f"- Best latency: {trend.best_latency_ms:.3f} ms",
        f"- Latest latency: {trend.latest_latency_ms:.3f} ms",
        f"- Improvement from baseline: {trend.improvement_pct:.2f}%",
        f"- Stop reason: {stop_reason}",
        "",
        "## Iteration Metrics",
        "",
        "| Iteration | Latency (ms) | Throughput (fps) | Power (W) | Memory (MB) |",
        "|---|---:|---:|---:|---:|",
    ]

    for idx, item in enumerate(history):
        lines.append(
            "| "
            f"{idx} | {item.latency_ms:.3f} | "
            f"{'' if item.throughput_fps is None else f'{item.throughput_fps:.3f}'} | "
            f"{'' if item.power_watts is None else f'{item.power_watts:.3f}'} | "
            f"{'' if item.memory_mb is None else f'{item.memory_mb:.3f}'} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
