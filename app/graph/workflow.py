# app/graph/workflow.py
from functools import lru_cache, partial
from langgraph.graph import StateGraph

from app.graph.state import ResearchState

from app.core.schemas import FinalResponse

from app.graph.edges import add_workflow_edges

from app.agents import *

from app.graph.nodes import planner_node, researcher_node, summarizer_node, judge_node
 

@lru_cache
def get_workflow():
    planner = PlannerAgent()
    researcher = ResearcherAgent()
    summarizer = SummariserAgent()
    judge = JudgeAgent()

    graph = StateGraph(ResearchState)

    graph.add_node(
        "planner",
        partial(planner_node, planner=planner),
    )
    graph.add_node(
        "researcher",
        partial(researcher_node, researcher=researcher),
    )
    graph.add_node(
        "summarizer",
        partial(summarizer_node, summarizer=summarizer),
    )
    graph.add_node(
        "judge",
        partial(judge_node, judge=judge),
    )

    add_workflow_edges(graph)
    return graph.compile()


def run_workflow(query: str) -> FinalResponse:
    """Run the end-to-end research workflow for a user query.

    Args:
        query (str): The user query to research.

    Returns:
        FinalResponse: Final synthesized answer converted from the workflow state.
    """

    final_state = get_workflow().invoke(
        {
            "query": query,
            "status": "started",
            "retry_count": 0,
        }
    )

    return FinalResponse.model_validate(final_state["synthesis"])
