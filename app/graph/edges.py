from langgraph.graph import END, START, StateGraph


def decide_next_step(state: dict) -> str:
    evaluation = state.get("evaluation") or {}
    retry_count = state.get("retry_count", 0)

    if retry_count >= 2:
        return "summarizer"

    if evaluation.get("passed") is True:
        return "summarizer"

    return "researcher"


def add_workflow_edges(graph: StateGraph) -> StateGraph:
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "judge")
    graph.add_conditional_edges(
        "judge",
        decide_next_step,
        {
            "researcher": "researcher",
            "summarizer": "summarizer",
        },
    )
    graph.add_edge("summarizer", END)

    return graph
