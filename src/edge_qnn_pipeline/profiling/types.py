from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileResult:
    latency_ms: float
    throughput_fps: float | None = None
    power_watts: float | None = None
    memory_mb: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
