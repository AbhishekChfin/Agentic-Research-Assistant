from langgraph.graph import END, START, StateGraph


def decide_next_step(state: dict) -> str:
    """
    Decide whether the graph should continue research or finish
    based on the judge agent's evaluation.

    Args:
        state: Current state of the graph.

    Returns:
        The name of the next graph node to execute.
    """
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
    """
    Connecting nodes using edges forming complete sequential workflow 
    """
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
