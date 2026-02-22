from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from edge_qnn_pipeline.config import TargetConfig


@dataclass(frozen=True)
class TargetProfile:
    name: str
    device_family: str
    soc: str
    qnn_backend: str
    htp_soc_id: int | None
    htp_arch: str | None
    notes: str


TARGETS = {
    "galaxy_s24_family": TargetProfile(
        name="galaxy_s24_family",
        device_family="Samsung Galaxy S24 (Family)",
        soc="SM8650",
        qnn_backend="htp",
        htp_soc_id=43,
        htp_arch="v75",
        notes="Primary target for Snapdragon 8 Gen 3 variants used in S24 family.",
    ),
    "snapdragon_8_elite_reference": TargetProfile(
        name="snapdragon_8_elite_reference",
        device_family="Snapdragon 8 Elite Reference",
        soc="SM8750",
        qnn_backend="htp",
        htp_soc_id=57,
        htp_arch="v79",
        notes="Reference profile for next-generation flagship SoCs.",
    ),
}


def get_target(name: str) -> TargetProfile:
    if name not in TARGETS:
        supported = ", ".join(sorted(TARGETS))
        raise KeyError(f"Unknown target '{name}'. Supported targets: {supported}")
    return TARGETS[name]


def list_targets() -> Iterable[TargetProfile]:
    return TARGETS.values()


def hydrate_target_config(target: TargetConfig) -> TargetConfig:
    if target.name not in TARGETS:
        return target

    preset = TARGETS[target.name]
    if not target.device_family:
        target.device_family = preset.device_family
    if not target.soc:
        target.soc = preset.soc
    if not target.qnn_backend:
        target.qnn_backend = preset.qnn_backend
    if target.htp_soc_id is None:
        target.htp_soc_id = preset.htp_soc_id
    if target.htp_arch is None:
        target.htp_arch = preset.htp_arch
    return target
