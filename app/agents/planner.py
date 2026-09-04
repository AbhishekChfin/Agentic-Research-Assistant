from functools import lru_cache
from pathlib import Path

from app.core.schemas import ResearchPlan
from app.models.planner_llm import get_planner_llm


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner.md"


@lru_cache(maxsize=128)
def _load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


class PlannerAgent:
    def __init__(self):
        self.llm = get_planner_llm()
        self.prompt_template = _load_prompt(str(PROMPT_PATH))

    def create_plan(self, query: str, validation_feedback: list[str]) -> ResearchPlan:
        feedback = "\n".join(validation_feedback or ["No previous validation errors."])

        prompt = self.prompt_template.format(
            query=query,
            validation_feedback=feedback)
        structured_llm = self.llm.with_structured_output(ResearchPlan) # structured_llm will return ResearchPlan schema so we can ignore the plance error below 
        response = structured_llm.invoke(prompt)

        if not isinstance(response, ResearchPlan):
            raise TypeError(
                f"Expected ResearchPlan, got {type(response).__name__}"
            )
        return response 


# if __name__ == "__main__":
#     planner = PlannerAgent()
#     plan = planner.create_plan("what are class objects in python.")
#     plan
