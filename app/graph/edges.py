from langgraph.graph import END, START, StateGraph


def decide_next_step(state: dict) -> str:
    evaluation = state.get("evaluation") or {}
    retry_count = state.get("retry_count", 0)
    recommended_action = evaluation.get("recommended_action", "stop_and_summarize")

    if evaluation.get("passed") is True:
        return "end"

    if retry_count >= 2:
        return "end"

    if recommended_action == "replan":
        return "planner"

    return "end"


def add_workflow_edges(graph: StateGraph) -> StateGraph:
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "summarizer")
    graph.add_edge("summarizer", "judge")
    graph.add_conditional_edges(
        "judge",
        decide_next_step,
        {
            "planner": "planner",
            "end": END,
        },
    )

    return graph
