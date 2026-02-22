from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from langchain_core.prompts import ChatPromptTemplate

from edge_qnn_pipeline.config import AgentConfig, OptimizationConfig
from edge_qnn_pipeline.profiling.types import ProfileResult


LOG = logging.getLogger(__name__)


@dataclass
class IterationAnalysis:
    summary: str
    suspected_bottleneck: str
    recommended_adjustment: str
    confidence: float


class IterationAnalyzer:
    def analyze(
        self,
        history: list[ProfileResult],
        target_latency_ms: float,
        last_action_name: str,
    ) -> IterationAnalysis:
        raise NotImplementedError


class RuleBasedIterationAnalyzer(IterationAnalyzer):
    def analyze(
        self,
        history: list[ProfileResult],
        target_latency_ms: float,
        last_action_name: str,
    ) -> IterationAnalysis:
        latest = history[-1]
        ratio = latest.latency_ms / target_latency_ms if target_latency_ms > 0 else 1.0

        if ratio > 1.5:
            summary = "Latency far above target."
            bottleneck = "Compute-bound graph sections likely dominate runtime."
            adjustment = "Prioritize aggressive quantization and input downscaling."
            confidence = 0.75
        elif ratio > 1.1:
            summary = "Latency above target but within optimization range."
            bottleneck = "Mixed operator precision or memory traffic overhead likely remains."
            adjustment = "Tune calibration/precision mix and backend extension parameters."
            confidence = 0.67
        else:
            summary = "Latency near target."
            bottleneck = "Minor scheduling or kernel-level overhead."
            adjustment = "Stabilize configuration and validate with repeated profiling."
            confidence = 0.6

        if len(history) >= 2:
            prev = history[-2].latency_ms
            if prev > 0:
                delta_pct = ((prev - latest.latency_ms) / prev) * 100.0
                if delta_pct < 1.0 and latest.latency_ms > target_latency_ms:
                    summary = "Recent change yielded minimal improvement."
                    bottleneck = "Optimization saturation at current architecture/shape."
                    adjustment = "Use architecture hook or stronger structural simplification."
                    confidence = 0.8

        summary = f"{summary} Last action: {last_action_name}."
        return IterationAnalysis(summary, bottleneck, adjustment, confidence)


class LLMIterationAnalyzer(IterationAnalyzer):
    def __init__(self, agent: AgentConfig):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "agent.mode='openai' requires langchain-openai. Install it first."
            ) from exc

        model = agent.model or "gpt-4.1-mini"
        self.llm = ChatOpenAI(model=model, temperature=agent.temperature, max_tokens=agent.max_tokens)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an edge AI performance engineer. Respond only as compact JSON with keys "
                    "summary,suspected_bottleneck,recommended_adjustment,confidence.",
                ),
                (
                    "human",
                    "Target latency (ms): {target_latency_ms}\n"
                    "Last action: {last_action_name}\n"
                    "History: {history_json}\n"
                    "Provide one-step analysis for next iteration.",
                ),
            ]
        )

    def analyze(
        self,
        history: list[ProfileResult],
        target_latency_ms: float,
        last_action_name: str,
    ) -> IterationAnalysis:
        history_json = json.dumps([asdict(h) for h in history])
        message = self.prompt.invoke(
            {
                "target_latency_ms": target_latency_ms,
                "last_action_name": last_action_name,
                "history_json": history_json,
            }
        )
        response = self.llm.invoke(message)
        raw = response.content if isinstance(response.content, str) else json.dumps(response.content)

        try:
            parsed = json.loads(raw)
            return IterationAnalysis(
                summary=parsed.get("summary", "LLM analysis generated."),
                suspected_bottleneck=parsed.get("suspected_bottleneck", "unspecified"),
                recommended_adjustment=parsed.get("recommended_adjustment", "unspecified"),
                confidence=float(parsed.get("confidence", 0.5)),
            )
        except Exception:
            LOG.warning("Failed to parse LLM analysis as JSON. Falling back to generic note.")
            return IterationAnalysis(
                summary="LLM analysis parse failure",
                suspected_bottleneck="unknown",
                recommended_adjustment="fallback to rule-based strategy",
                confidence=0.2,
            )


def build_iteration_analyzer(agent: AgentConfig, optimization: OptimizationConfig) -> IterationAnalyzer:
    if not agent.enabled or agent.mode == "rule_based":
        return RuleBasedIterationAnalyzer()
    if agent.mode == "openai":
        return LLMIterationAnalyzer(agent)
    raise ValueError(f"Unsupported agent.mode '{agent.mode}'")
