from edge_qnn_pipeline.profiling.aihub_client import (
    QAIHubSDKProfilerBackend,
    _coerce_float,
    _flatten_metrics,
)


def test_coerce_float_time_units():
    assert _coerce_float("4.5 ms", unit="time") == 4.5
    assert _coerce_float("2 s", unit="time") == 2000.0
    assert _coerce_float("1200 us", unit="time") == 1.2


def test_flatten_and_extract_keywords():
    payload = {
        "summary": {
            "inference_time_ms": "7.8 ms",
            "throughput": "128.2 fps",
            "power": {"avg_power_w": "2.1 W"},
        },
        "memory": {"peak_memory_mb": "512"},
    }

    flattened = _flatten_metrics(payload)
    assert "summary.inference_time_ms" in flattened

    latency = QAIHubSDKProfilerBackend._extract_required_metric(payload, ("inference_time",), unit="time")
    throughput = QAIHubSDKProfilerBackend._extract_optional_metric(payload, ("throughput", "fps"))
    power = QAIHubSDKProfilerBackend._extract_optional_metric(payload, ("power", "watt"))
    memory = QAIHubSDKProfilerBackend._extract_optional_metric(payload, ("memory", "ram"))

    assert latency == 7.8
    assert throughput == 128.2
    assert power == 2.1
    assert memory == 512.0
