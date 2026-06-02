from app.core.schemas import EvidenceItem, FinalResponse, ResearchPlan, Subtask
from app.synthesis.summarizer import synthesize
from app.models.answer_llm import get_answer_llm

# class FakeAnswerLLM:
#     def invoke(self, prompt: str):
#         return "LangGraph supports graph-based workflows based on the provided evidence."


def test_synthesize_returns_final_response():
    plan = ResearchPlan(
        query="Compare LangGraph and CrewAI.",
        subtasks=[
            Subtask(
                id="s1",
                task="Find public framework details.",
                mode="web",
            )
        ],
    )
    findings = [
        EvidenceItem(
            claim="LangGraph supports graph-based workflows.",
            source_title="Dummy Web Source",
            source_url="https://example.com/web",
            confidence=0.5,
        )
    ]

    response = synthesize(plan, findings, llm=get_answer_llm())

    assert isinstance(response, FinalResponse)
    assert response.plan == plan
    assert response.findings == findings
    assert response.status == "completed"
    assert response.response
    assert "LangGraph" in response.response
