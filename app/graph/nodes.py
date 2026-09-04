from app.agents import JudgeAgent, PlannerAgent, ResearcherAgent, SummariserAgent
from app.core.schemas import EvidenceItem, EvaluationResult, ResearchPlan
from app.graph.state import ResearchState


def validate_plan_node(state: ResearchState) -> ResearchState:
    plan = ResearchPlan.model_validate(state["plan"]) # type: ignore

    errors = []

    if not plan.subtasks:
        errors.append("Plan has no subtasks.")

    if len(plan.subtasks) > 4:
        errors.append("Plan exceeds the maximum subtask budget.")

    ids = [subtask.id for subtask in plan.subtasks]
    if len(ids) != len(set(ids)):
        errors.append("Subtask IDs must be unique.")

    return {
        "plan_valid": not errors,
        "plan_validation_errors": errors,
    }

def planner_node(state: ResearchState, planner: PlannerAgent) -> ResearchState:
    """
    Using planner agent to create plan
    """
    feedback = state.get("plan_validation_errors", [])
    plan = planner.create_plan(state["query"], validation_feedback = feedback) # type: ignore

    return {
        "plan": plan.model_dump(),
        "subtasks": [subtask.model_dump() for subtask in plan.subtasks],
        "status": "planned",
    }


def researcher_node(state: ResearchState, researcher: ResearcherAgent) -> ResearchState:
    """
    Excuting research using the plan stored in workflow state
    Args:
        Updated state 
    Return:
        Updated graph state using collected evidence and completed subtasks
    """
    plan = ResearchPlan.model_validate(state["plan"]) # type: ignore
    findings = researcher.run(plan)

    return {
        "evidence": [finding.model_dump() for finding in findings],
        "completed_subtasks": state.get("subtasks", []),
        "status": "researched",
    }


def judge_node(state: ResearchState, judge: JudgeAgent) -> ResearchState:
    """
    Evaluating the final answer against the research plan and evidence.
    Args:
        state: Current workflow state containing the query, plan,
            evidence, and final answer.

    Returns:
        Updated workflow state containing the evaluation result,
        updated retry count, and judged status.
    """

    plan = ResearchPlan.model_validate(state["plan"]) # type: ignore
    findings = [
        EvidenceItem.model_validate(item)
        for item in state.get("evidence", [])
    ]
    final_answer = state.get("final_response", "")

    
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


def summarizer_node(state: ResearchState, summarizer: SummariserAgent) -> ResearchState:
    """
    Generate a final response by synthesizing the research plan and collected evidence.
    """
    plan = ResearchPlan.model_validate(state["plan"]) # type: ignore
    findings = [
        EvidenceItem.model_validate(item)
        for item in state.get("evidence", [])
    ]
    final_response = summarizer.synthesize(plan, findings)

    return {
        "synthesis": final_response.model_dump(),
        "final_response": final_response.response,
        "status": "completed",
    }