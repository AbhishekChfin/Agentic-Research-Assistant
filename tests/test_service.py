from app.api import service
from app.core.schemas import FinalResponse


def test_run_research_returns_valid_final_response():
    response = service.run_research("Compare public and internal RAG guidance.")

    assert isinstance(response, FinalResponse)
    assert response.plan.query
    assert response.status == "completed"
    assert 2 <= len(response.plan.subtasks) <= 5
    assert len(response.findings) == len(response.plan.subtasks)
    assert response.response
