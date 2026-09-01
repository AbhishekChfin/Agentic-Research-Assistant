from pathlib import Path

from app.core.schemas import ResearchPlan
from app.models.planner_llm import get_planner_llm


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner.md"


class PlannerAgent:
    def __init__(self):
        self.llm = get_planner_llm() 
        self.prompt_template = PROMPT_PATH.read_text()

    def create_plan(self, query: str) -> ResearchPlan:
        prompt = self.prompt_template.format(query=query)
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
#     print(plan)
#     print(plan.model_dump().subtasks[0].task)
