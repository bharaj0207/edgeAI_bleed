from pathlib import Path

import pytest

from edge_qnn_pipeline.config import AIHubConfig, HardwareConfig, load_config
from edge_qnn_pipeline.profiling.aihub_client import (
    PhysicalDeviceProfilerBackend,
    _extract_latency_from_qnn_net_run,
    build_profiler,
)


def test_extract_latency_from_qnn_output():
    output = "Average latency: 12.34 ms\nThroughput: 80.1 fps"
    assert _extract_latency_from_qnn_net_run(output) == 12.34


def test_extract_latency_from_qnn_output_seconds_unit():
    output = "Inference time: 0.02 s\nDone"
    assert _extract_latency_from_qnn_net_run(output) == 20.0


def test_load_config_hardware(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  framework: onnx
  model_path: ./m.onnx
target:
  name: t
  soc: s
  device_family: d
qnn:
  sdk_root: /tmp/qnn
optimization:
  target_latency_ms: 10
aihub:
  mode: mock
  hardware:
    ip: 10.0.0.2
    pem_key: /keys/device.pem
    qnn_net_run_path: /opt/qnn/qnn-net-run
    server_ip: 10.0.0.10
    server_user: ubuntu
    server_pem_key: /keys/server.pem
"""
    )

    parsed = load_config(cfg)
    assert parsed.aihub.hardware is not None
    assert parsed.aihub.hardware.ip == "10.0.0.2"
    assert parsed.aihub.hardware.user == "root"
    assert parsed.aihub.hardware.server_ip == "10.0.0.10"


def test_build_profiler_prefers_physical_when_hardware_present(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  framework: onnx
  model_path: ./m.onnx
target:
  name: t
  soc: s
  device_family: d
qnn:
  sdk_root: /tmp/qnn
optimization:
  target_latency_ms: 10
aihub:
  mode: mock
  hardware:
    ip: 10.0.0.2
    pem_key: /keys/device.pem
    qnn_net_run_path: /opt/qnn/qnn-net-run
"""
    )

    parsed = load_config(cfg)
    profiler = build_profiler(parsed.aihub)
    assert isinstance(profiler, PhysicalDeviceProfilerBackend)


def test_physical_backend_requires_server_user_when_server_ip_is_set():
    backend = PhysicalDeviceProfilerBackend(
        AIHubConfig(
            mode="mock",
            hardware=HardwareConfig(
                ip="10.0.0.2",
                pem_key=Path("/keys/device.pem"),
                qnn_net_run_path="/opt/qnn/qnn-net-run",
                server_ip="10.0.0.10",
            ),
        )
    )
    with pytest.raises(ValueError, match="server_user"):
        backend._base_ssh_cmd()
