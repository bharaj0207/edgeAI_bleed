from edge_qnn_pipeline.command import CommandRunner
from edge_qnn_pipeline.config import ModelConfig, OptimizationConfig, QuantizationConfig
from edge_qnn_pipeline.optimization.strategy import OptimizationPlanner
from edge_qnn_pipeline.profiling.types import ProfileResult


def test_stop_when_target_met(tmp_path):
    planner = OptimizationPlanner(
        optimization=OptimizationConfig(target_latency_ms=5.0),
        model=ModelConfig(framework="onnx", model_path=tmp_path / "m.onnx", input_shape=[1, 3, 224, 224]),
        quantization=QuantizationConfig(),
        runner=CommandRunner(dry_run=True),
    )

    stop, _ = planner.should_stop(ProfileResult(latency_ms=4.2), iteration=0)
    assert stop


def test_resize_action_respects_min_edge(tmp_path):
    planner = OptimizationPlanner(
        optimization=OptimizationConfig(target_latency_ms=5.0, min_input_edge=160),
        model=ModelConfig(framework="onnx", model_path=tmp_path / "m.onnx", input_shape=[1, 3, 180, 180]),
        quantization=QuantizationConfig(),
        runner=CommandRunner(dry_run=True),
    )

    action = planner.propose_action([ProfileResult(latency_ms=10.0)], iteration=3)
    assert action is not None
    assert action.updates["model.input_shape"][-1] >= 160
