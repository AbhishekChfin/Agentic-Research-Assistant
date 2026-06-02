from pathlib import Path

from app.core.schemas import ResearchPlan
from app.models.planner_llm import get_planner_llm


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner.md"


class PlannerAgent:
    def __init__(self):
        self.llm = get_planner_llm()
        self.prompt_template = PROMPT_PATH.read_text()

    def create_plan(self, query: str) -> ResearchPlan:
        prompt = self._build_prompt(query)
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        json_text = self._extract_json(content)
        return ResearchPlan.model_validate_json(json_text)

    def _build_prompt(self, query: str) -> str:
        schema = ResearchPlan.model_json_schema()
        return self.prompt_template.format(
            schema=schema,
            query=query
        )

    def _extract_json(self, content: str) -> str:
        content = content.strip()

        if content.startswith("```json"):
            content = content.removeprefix("```json").removesuffix("```").strip()
        elif content.startswith("```"):
            content = content.removeprefix("```").removesuffix("```").strip()

        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(f"Planner did not return JSON: {content}")

        return content[start : end + 1]


if __name__ == "__main__":
    planner = PlannerAgent()
    plan = planner.create_plan("Compare LangGraph, AutoGen, and CrewAI for production multi-agent workflows.")
    print(plan.model_dump_json(indent=2))
