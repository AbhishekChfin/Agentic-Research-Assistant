from app.core.schemas import FinalResponse
from app.graph.workflow import run_workflow


def test_run_workflow_returns_final_response():
    response = run_workflow("Compare public and internal RAG guidance.")

    assert isinstance(response, FinalResponse)
    assert response.plan.query
    assert response.status == "completed"
    assert 2 <= len(response.plan.subtasks) <= 5
    assert len(response.findings) == len(response.plan.subtasks)
    assert response.response