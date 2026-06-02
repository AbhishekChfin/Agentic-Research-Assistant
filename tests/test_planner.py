from app.agents.planner import PlannerAgent
from app.core.schemas import ResearchPlan


QUESTIONS = [
    "What are the latest major releases in LangGraph?",
    "What do our internal notes say about vector databases?",
    "Compare our internal RAG notes with current public best practices.",
    "Tell me what we know about Gemini.",
    "Break down the research needed to decide whether to build or buy an enterprise research assistant.",
]


def test_planner_returns_valid_research_plan():
    planner = PlannerAgent()

    for question in QUESTIONS:
        plan = planner.create_plan(question)

        assert isinstance(plan, ResearchPlan)
        assert plan.query
        assert 2 <= len(plan.subtasks) <= 5

        for subtask in plan.subtasks:
            assert subtask.id
            assert subtask.task
            assert subtask.mode in {"web", "rag"}
