from langgraph.graph import StateGraph

from app.core.schemas import FinalResponse
from app.graph.edges import add_workflow_edges
from app.graph.nodes import planner_node, researcher_node, summarizer_node
from app.graph.state import ResearchState


def build_workflow():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("summarizer", summarizer_node)

    add_workflow_edges(graph)

    return graph.compile()


def run_workflow(query: str) -> FinalResponse:
    workflow = build_workflow()

    final_state = workflow.invoke(
        {
            "query": query,
            "status": "started",
        }
    )

    return FinalResponse.model_validate(final_state["synthesis"])
