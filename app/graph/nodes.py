from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.core.schemas import EvidenceItem, EvaluationResult, ResearchPlan
from app.agents.judge import JudgeAgent
from app.graph.state import ResearchState
from app.agents.summarizer import SummariserAgent


def planner_node(state: ResearchState) -> ResearchState:
    planner = PlannerAgent()
    plan = planner.create_plan(state["query"]) # type: ignore

    return {
        "plan": plan.model_dump(),
        "subtasks": [subtask.model_dump() for subtask in plan.subtasks],
        "status": "planned",
    }


def researcher_node(state: ResearchState) -> ResearchState:
    plan = ResearchPlan.model_validate(state["plan"]) # type: ignore
    researcher = ResearcherAgent()
    findings = researcher.run(plan)

    return {
        "evidence": [finding.model_dump() for finding in findings],
        "completed_subtasks": state.get("subtasks", []),
        "status": "researched",
    }


def judge_node(state: ResearchState) -> ResearchState:
    plan = ResearchPlan.model_validate(state["plan"]) # type: ignore
    findings = [
        EvidenceItem.model_validate(item)
        for item in state.get("evidence", [])
    ]
    final_answer = state.get("final_response", "")

    judge = JudgeAgent()
    evaluation = judge.evaluate_final_answer(
        query=state["query"], # type: ignore
        answer=final_answer,
        plan=plan,
        findings=findings,
    )

    retry_count = state.get("retry_count", 0)

    if not evaluation.passed:
        retry_count += 1

    return {
        "evaluation": evaluation.model_dump(),
        "retry_count": retry_count,
        "status": "judged",
    }


def summarizer_node(state: ResearchState) -> ResearchState:
    plan = ResearchPlan.model_validate(state["plan"]) # type: ignore
    findings = [
        EvidenceItem.model_validate(item)
        for item in state.get("evidence", [])
    ]
    summarise = SummariserAgent()
    final_response = summarise.synthesize(plan, findings)

    return {
        "synthesis": final_response.model_dump(),
        "final_response": final_response.response,
        "status": "completed",
    }