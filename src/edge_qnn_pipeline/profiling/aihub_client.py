from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol

import requests

from edge_qnn_pipeline.config import AIHubConfig, TargetConfig
from edge_qnn_pipeline.profiling.types import ProfileResult


LOG = logging.getLogger(__name__)


class ProfilerBackend(Protocol):
    def profile(self, context_binary: Path, target: TargetConfig, iteration: int) -> ProfileResult:
        ...


class MockProfilerBackend:
    def profile(self, context_binary: Path, target: TargetConfig, iteration: int) -> ProfileResult:
        # Deterministic synthetic metric for development without cloud calls.
        payload = f"{context_binary}:{target.soc}:{iteration}".encode("utf-8")
        seed = int(hashlib.sha256(payload).hexdigest()[:8], 16)
        baseline = 5.0 + (seed % 4000) / 200.0
        improvement = 0.85 ** iteration
        latency = max(1.0, baseline * improvement)
        throughput = 1000.0 / latency
        return ProfileResult(
            latency_ms=latency,
            throughput_fps=throughput,
            raw={"mode": "mock", "seed": seed, "target": target.name},
        )


class QAIHubProfilerBackend:
    def __init__(self, config: AIHubConfig):
        if not config.base_url:
            raise ValueError("aihub.base_url is required when aihub.mode is 'qai_hub'")
        self.config = config
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key in environment variable '{config.api_key_env}'."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def profile(self, context_binary: Path, target: TargetConfig, iteration: int) -> ProfileResult:
        artifact_id = self._upload_artifact(context_binary)
        job_id = self._create_profile_job(artifact_id, target, iteration)
        job = self._wait_for_completion(job_id)
        metrics = self._extract_metrics(job)
        latency = self._extract_latency(metrics)
        throughput = self._extract_optional_metric(metrics, ["throughput_fps", "fps"])
        power = self._extract_optional_metric(metrics, ["power_watts", "avg_power_w"])
        memory = self._extract_optional_metric(metrics, ["memory_mb", "peak_memory_mb"])
        return ProfileResult(
            latency_ms=latency,
            throughput_fps=throughput,
            power_watts=power,
            memory_mb=memory,
            raw={"job": job, "metrics": metrics},
        )

    def _upload_artifact(self, path: Path) -> str:
        url = f"{self.config.base_url.rstrip('/')}/v1/artifacts"
        with path.open("rb") as handle:
            response = self.session.post(url, files={"file": handle}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        artifact_id = payload.get("id") or payload.get("artifact_id")
        if not artifact_id:
            raise RuntimeError(f"Unexpected artifact upload response: {payload}")
        LOG.info("Uploaded artifact %s", artifact_id)
        return artifact_id

    def _create_profile_job(self, artifact_id: str, target: TargetConfig, iteration: int) -> str:
        url = f"{self.config.base_url.rstrip('/')}/v1/jobs"
        payload = {
            "type": "qnn_profile",
            "project": self.config.project,
            "target": {
                "name": target.name,
                "device_family": target.device_family,
                "soc": target.soc,
            },
            "inputs": {
                "context_binary_artifact_id": artifact_id,
            },
            "metadata": {
                "iteration": iteration,
            },
        }
        response = self.session.post(url, json=payload, timeout=120)
        response.raise_for_status()
        content = response.json()
        job_id = content.get("id") or content.get("job_id")
        if not job_id:
            raise RuntimeError(f"Unexpected job creation response: {content}")
        LOG.info("Created AI Hub job %s", job_id)
        return job_id

    def _wait_for_completion(self, job_id: str) -> dict[str, Any]:
        deadline = time.time() + self.config.timeout_sec
        url = f"{self.config.base_url.rstrip('/')}/v1/jobs/{job_id}"

        while True:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            status = (payload.get("status") or "").lower()
            if status in {"completed", "success", "succeeded"}:
                return payload
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(f"AI Hub job {job_id} failed with payload: {payload}")
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for AI Hub job {job_id}")
            time.sleep(self.config.polling_interval_sec)

    def _extract_metrics(self, job_payload: dict[str, Any]) -> dict[str, Any]:
        metrics = job_payload.get("metrics")
        if isinstance(metrics, dict):
            return metrics

        job_id = job_payload.get("id") or job_payload.get("job_id")
        if not job_id:
            raise RuntimeError("Job payload missing id and metrics")

        url = f"{self.config.base_url.rstrip('/')}/v1/jobs/{job_id}/metrics"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected metrics response: {payload}")
        return payload

    def _extract_latency(self, metrics: dict[str, Any]) -> float:
        value = self._extract_optional_metric(metrics, ["latency_ms", "avg_latency_ms", "inference_time_ms"])
        if value is None:
            raise RuntimeError(f"Could not locate latency in metrics payload: {metrics}")
        return float(value)

    @staticmethod
    def _extract_optional_metric(metrics: dict[str, Any], keys: list[str]) -> float | None:
        for key in keys:
            if key in metrics and metrics[key] is not None:
                return float(metrics[key])
        return None


def build_profiler(config: AIHubConfig) -> ProfilerBackend:
    if config.mode == "mock":
        return MockProfilerBackend()
    if config.mode == "qai_hub":
        return QAIHubProfilerBackend(config)
    raise ValueError(f"Unsupported aihub.mode '{config.mode}'")
