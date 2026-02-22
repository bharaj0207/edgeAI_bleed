from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

import requests
import yaml

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


class QAIHubRestProfilerBackend:
    def __init__(self, config: AIHubConfig):
        if not config.base_url:
            raise ValueError("aihub.base_url is required when aihub.mode is 'qai_hub_rest'")
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
            raw={"mode": "qai_hub_rest", "job": job, "metrics": metrics},
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
        value = self._extract_optional_metric(
            metrics,
            ["latency_ms", "avg_latency_ms", "inference_time_ms"],
        )
        if value is None:
            raise RuntimeError(f"Could not locate latency in metrics payload: {metrics}")
        return float(value)

    @staticmethod
    def _extract_optional_metric(metrics: dict[str, Any], keys: list[str]) -> float | None:
        for key in keys:
            if key in metrics and metrics[key] is not None:
                return float(metrics[key])
        return None


class QAIHubSDKProfilerBackend:
    def __init__(self, config: AIHubConfig):
        self.config = config
        try:
            import qai_hub as hub
        except ImportError as exc:
            raise RuntimeError(
                "aihub.mode='qai_hub_sdk' requires the qai-hub package. "
                "Install with: pip install -e '.[qaihub]'"
            ) from exc

        self.hub = hub
        api_key = os.getenv(config.api_key_env)
        if api_key and hasattr(self.hub, "set_session_token"):
            self.hub.set_session_token(api_key)

    def profile(self, context_binary: Path, target: TargetConfig, iteration: int) -> ProfileResult:
        if not context_binary.exists():
            raise FileNotFoundError(f"Context binary not found: {context_binary}")

        device_name = self.config.device or target.device_family or target.name
        device = self.hub.Device(device_name)

        model = self.hub.upload_model(str(context_binary))
        options = self._build_options()
        job_name = f"{target.name}-iter-{iteration:02d}"

        LOG.info("Submitting QAI Hub profile job on device '%s' with options: %s", device_name, options)
        profile_job = self.hub.submit_profile_job(
            model=model,
            device=device,
            name=job_name,
            options=options,
        )

        status = profile_job.wait(timeout=self.config.timeout_sec)
        if str(status).upper() != "SUCCESS":
            raise RuntimeError(
                f"QAI Hub profile job failed with status {status}. "
                f"Job URL: {getattr(profile_job, 'url', 'n/a')}"
            )

        profile_payload = profile_job.download_profile()
        metrics = self._normalize_profile_payload(profile_payload)
        latency = self._extract_required_metric(metrics, LATENCY_KEYWORDS, unit="time")
        throughput = self._extract_optional_metric(metrics, THROUGHPUT_KEYWORDS)
        power = self._extract_optional_metric(metrics, POWER_KEYWORDS)
        memory = self._extract_optional_metric(metrics, MEMORY_KEYWORDS)

        return ProfileResult(
            latency_ms=latency,
            throughput_fps=throughput,
            power_watts=power,
            memory_mb=memory,
            raw={
                "mode": "qai_hub_sdk",
                "status": str(status),
                "job_url": getattr(profile_job, "url", None),
                "device": device_name,
                "options": options,
                "metrics": metrics,
            },
        )

    def _build_options(self) -> str:
        options = (self.config.profile_options or "").strip()
        if "--target_runtime" not in options:
            options = f"{options} --target_runtime qnn_context_binary".strip()
        return options

    def _normalize_profile_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload

        if isinstance(payload, Path):
            return self._read_profile_file(payload)

        if isinstance(payload, str):
            candidate = payload.strip()
            if candidate.startswith("{") or candidate.startswith("["):
                try:
                    loaded = json.loads(candidate)
                    if isinstance(loaded, dict):
                        return loaded
                except json.JSONDecodeError:
                    pass

            possible_path = Path(candidate)
            if possible_path.exists():
                return self._read_profile_file(possible_path)

            raise RuntimeError(f"Unsupported profile payload string: {payload[:120]}")

        raise RuntimeError(f"Unsupported profile payload type: {type(payload)}")

    @staticmethod
    def _read_profile_file(path: Path) -> dict[str, Any]:
        text = path.read_text()
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, dict):
                return parsed_json
        except json.JSONDecodeError:
            pass

        parsed_yaml = yaml.safe_load(text)
        if isinstance(parsed_yaml, dict):
            return parsed_yaml
        raise RuntimeError(f"Profile file does not contain a dict payload: {path}")

    @staticmethod
    def _extract_required_metric(metrics: dict[str, Any], keywords: tuple[str, ...], unit: str | None = None) -> float:
        value = QAIHubSDKProfilerBackend._extract_optional_metric(metrics, keywords, unit=unit)
        if value is None:
            raise RuntimeError(
                "Could not determine latency from profile payload. "
                f"Expected one of keywords {keywords}. Payload keys: {list(metrics.keys())}"
            )
        return value

    @staticmethod
    def _extract_optional_metric(
        metrics: dict[str, Any],
        keywords: tuple[str, ...],
        unit: str | None = None,
    ) -> float | None:
        flattened = _flatten_metrics(metrics)
        for key, raw_value in flattened.items():
            key_norm = _normalize_key(key)
            if any(keyword in key_norm for keyword in keywords):
                parsed = _coerce_float(raw_value, unit=unit)
                if parsed is not None:
                    return parsed
        return None


LATENCY_KEYWORDS = (
    "latency",
    "inference_time",
    "execution_time",
    "total_time",
)
THROUGHPUT_KEYWORDS = (
    "throughput",
    "fps",
    "frames_per_second",
)
POWER_KEYWORDS = (
    "power",
    "watt",
)
MEMORY_KEYWORDS = (
    "memory",
    "ram",
    "peak_mem",
)


def _flatten_metrics(value: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    if out is None:
        out = {}

    if isinstance(value, dict):
        for key, child in value.items():
            key_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_metrics(child, key_prefix, out)
        return out

    if isinstance(value, list):
        for idx, child in enumerate(value):
            key_prefix = f"{prefix}[{idx}]"
            _flatten_metrics(child, key_prefix, out)
        return out

    if prefix:
        out[prefix] = value
    return out


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _coerce_float(value: Any, unit: str | None = None) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        if unit == "time":
            return number
        return number

    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None

        match = re.search(r"([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([a-zA-Z]+)?", text)
        if not match:
            return None

        number = float(match.group(1))
        suffix = (match.group(2) or "").lower()

        if unit == "time":
            if suffix in {"s", "sec", "secs", "second", "seconds"}:
                return number * 1000.0
            if suffix in {"us", "usec", "microsecond", "microseconds"}:
                return number / 1000.0
            if suffix in {"ns", "nsec"}:
                return number / 1_000_000.0
            return number

        return number

    return None


def build_profiler(config: AIHubConfig) -> ProfilerBackend:
    if config.mode == "mock":
        return MockProfilerBackend()
    if config.mode == "qai_hub_rest":
        return QAIHubRestProfilerBackend(config)
    if config.mode in {"qai_hub", "qai_hub_sdk"}:
        return QAIHubSDKProfilerBackend(config)
    raise ValueError(f"Unsupported aihub.mode '{config.mode}'")
