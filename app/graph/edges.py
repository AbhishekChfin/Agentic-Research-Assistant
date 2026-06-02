from langgraph.graph import END, START, StateGraph


def add_workflow_edges(graph: StateGraph) -> StateGraph:
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "summarizer")
    graph.add_edge("summarizer", END)

    return graph
