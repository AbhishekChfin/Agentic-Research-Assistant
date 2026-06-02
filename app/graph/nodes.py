from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.core.schemas import EvidenceItem, ResearchPlan
from app.graph.state import ResearchState
from app.synthesis.summarizer import synthesize


def planner_node(state: ResearchState) -> ResearchState:
    planner = PlannerAgent()
    plan = planner.create_plan(state["query"])

    return {
        "plan": plan.model_dump(),
        "subtasks": [subtask.model_dump() for subtask in plan.subtasks],
        "status": "planned",
    }


def researcher_node(state: ResearchState) -> ResearchState:
    plan = ResearchPlan.model_validate(state["plan"])

    researcher = ResearcherAgent()
    findings = researcher.run(plan)

    return {
        "evidence": [finding.model_dump() for finding in findings],
        "completed_subtasks": state.get("subtasks", []),
        "status": "researched",
    }


def summarizer_node(state: ResearchState) -> ResearchState:
    plan = ResearchPlan.model_validate(state["plan"])
    findings = [
        EvidenceItem.model_validate(item)
        for item in state["evidence"]
    ]

    final_response = synthesize(plan, findings)

    return {
        "synthesis": final_response.model_dump(),
        "final_response": final_response.response,
        "status": "completed",
    }