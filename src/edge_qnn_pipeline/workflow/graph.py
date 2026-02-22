from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from edge_qnn_pipeline.workflow.nodes import PipelineNodes
from edge_qnn_pipeline.workflow.state import PipelineState


class PipelineGraph:
    def __init__(self, nodes: PipelineNodes):
        self.nodes = nodes

    def compile(self):
        graph = StateGraph(PipelineState)
        graph.add_node("initialize", self.nodes.initialize)
        graph.add_node("prepare_model", self.nodes.prepare_model)
        graph.add_node("select_action", self.nodes.select_action)
        graph.add_node("apply_action", self.nodes.apply_action)
        graph.add_node("convert_model", self.nodes.convert_model)
        graph.add_node("profile_model", self.nodes.profile_model)
        graph.add_node("analyze_profile", self.nodes.analyze_profile)
        graph.add_node("evaluate", self.nodes.evaluate)
        graph.add_node("finalize", self.nodes.finalize)

        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "prepare_model")
        graph.add_edge("prepare_model", "select_action")

        graph.add_conditional_edges(
            "select_action",
            self._route_after_select,
            {
                "apply": "apply_action",
                "finalize": "finalize",
            },
        )

        graph.add_edge("apply_action", "convert_model")
        graph.add_edge("convert_model", "profile_model")
        graph.add_edge("profile_model", "analyze_profile")
        graph.add_edge("analyze_profile", "evaluate")

        graph.add_conditional_edges(
            "evaluate",
            self._route_after_evaluate,
            {
                "loop": "select_action",
                "finalize": "finalize",
            },
        )

        graph.add_edge("finalize", END)
        return graph.compile()

    @staticmethod
    def _route_after_select(state: PipelineState) -> str:
        return "apply" if state.get("current_action") is not None else "finalize"

    @staticmethod
    def _route_after_evaluate(state: PipelineState) -> str:
        return "loop" if state.get("continue_loop", False) else "finalize"
