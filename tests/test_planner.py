from pathlib import Path

from app.agents import judge as judge_module
from app.agents import planner as planner_module
from app.agents.planner import PlannerAgent
from app.core.schemas import ResearchPlan


QUESTIONS = [
    "What are the latest major releases in LangGraph?",
    "What do our internal notes say about vector databases?",
    "Compare our internal RAG notes with current public best practices.",
    "Tell me what we know about Gemini.",
    "Break down the research needed to decide whether to build or buy an enterprise research assistant.",
]


def test_prompt_files_are_cached(monkeypatch):
    calls = {"count": 0}
    original_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        if str(self).endswith(("planner.md", "rubrics.md")):
            calls["count"] += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    planner_module._load_prompt(str(planner_module.PROMPT_PATH))
    planner_module._load_prompt(str(planner_module.PROMPT_PATH))
    judge_module._load_prompt(str(judge_module.PROMPT_PATH))
    judge_module._load_prompt(str(judge_module.PROMPT_PATH))

    assert calls["count"] == 2


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
