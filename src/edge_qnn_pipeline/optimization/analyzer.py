from __future__ import annotations

from dataclasses import dataclass

from edge_qnn_pipeline.profiling.types import ProfileResult


@dataclass
class TrendSummary:
    best_latency_ms: float
    latest_latency_ms: float
    improvement_pct: float


def summarize_trend(history: list[ProfileResult]) -> TrendSummary:
    if not history:
        return TrendSummary(best_latency_ms=0.0, latest_latency_ms=0.0, improvement_pct=0.0)

    first = history[0].latency_ms
    best = min(item.latency_ms for item in history)
    latest = history[-1].latency_ms

    improvement = 0.0
    if first > 0:
        improvement = ((first - best) / first) * 100.0

    return TrendSummary(best_latency_ms=best, latest_latency_ms=latest, improvement_pct=improvement)
