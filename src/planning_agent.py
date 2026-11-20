"""
planning_agent.py — Planner + orchestrator for the research workflow.

The planner decomposes a user query into subtasks, then the executor
runs each subtask through the appropriate tool (web search or RAG).
Finally the results are synthesized and evaluated.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import Literal

from src.agents import researcher_agent, summarizer_agent, judge_agent
from src.research_tools import web_search, rag_search

load_dotenv()

# ── Schemas ─────────────────────────────────────────────────────────

class Subtask(BaseModel):
    id: str
    task: str
    mode: Literal["web", "rag"]
    success_criteria: str | None = None


class ResearchPlan(BaseModel):
    query: str
    subtasks: list[Subtask]


class Evidence(BaseModel):
    claim: str
    source_title: str
    source_url: str
    confidence: float


class EvidenceItem(Evidence):
    subtask_id: str


class FinalResponse(BaseModel):
    plan: ResearchPlan
    findings: list[EvidenceItem]
    response: str
    status: str


class EvaluationResult(BaseModel):
    passed: bool
    score: float
    summary: str
    issues: list[str] = []
    missing_evidence: list[str] = []
    confidence: float
    notes: str | None = None


# ── Prompts ─────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are the Planner agent for a research assistant.

Your job:
- Read the user's research query.
- Break it into 1 to 4 clear subtasks.
- Choose one retrieval mode for each subtask:
  - "web" for fresh, public, external, or source-sensitive facts.
  - "rag" for internal documents, uploaded files, or domain-specific knowledge.
- Do not answer the user's query.

User query:
{query}

Previous plan validation feedback:
{validation_feedback}

If validation feedback is present, create a corrected plan that fixes every issue."""


# ── LLM helpers ─────────────────────────────────────────────────────

def _get_planner_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )


def _get_answer_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )


# ── Planner agent ───────────────────────────────────────────────────

def planner_agent(query: str, validation_feedback: list[str] | None = None) -> ResearchPlan:
    """Break a user query into a structured research plan."""
    llm = _get_planner_llm()
    feedback = "\n".join(validation_feedback or ["No previous validation errors."])
    prompt = PLANNER_PROMPT.format(query=query, validation_feedback=feedback)

    structured_llm = llm.with_structured_output(ResearchPlan)
    response = structured_llm.invoke(prompt)

    if not isinstance(response, ResearchPlan):
        raise TypeError(f"Expected ResearchPlan, got {type(response).__name__}")
    return response


# ── Plan validation ─────────────────────────────────────────────────

def validate_plan(plan: ResearchPlan) -> tuple[bool, list[str]]:
    """Validate a research plan. Returns (is_valid, errors)."""
    errors: list[str] = []

    if not plan.subtasks:
        errors.append("Plan has no subtasks.")
    if len(plan.subtasks) > 4:
        errors.append("Plan exceeds the maximum subtask budget.")

    ids = [s.id for s in plan.subtasks]
    if len(ids) != len(set(ids)):
        errors.append("Subtask IDs must be unique.")

    return (not errors, errors)


# ── Executor step ───────────────────────────────────────────────────

def executor_agent_step(plan: ResearchPlan) -> list[EvidenceItem]:
    """Execute each subtask in the plan through the appropriate tool."""
    return researcher_agent(plan)


# ── Full workflow ───────────────────────────────────────────────────

MAX_RETRIES = 2


def run_research_workflow(query: str) -> dict:
    """
    Run the end-to-end research pipeline:
        plan → validate → research → summarize → judge → (retry if needed)

    Returns a dict with plan, findings, response, and evaluation.
    """
    validation_feedback: list[str] = []

    # 1. Plan (with retry on validation failure)
    for _ in range(MAX_RETRIES + 1):
        plan = planner_agent(query, validation_feedback)
        is_valid, errors = validate_plan(plan)
        if is_valid:
            break
        validation_feedback = errors
    else:
        return {"status": "failed", "error": "Could not produce a valid plan."}

    # 2. Research
    findings = executor_agent_step(plan)

    # 3. Summarize → Judge loop
    for attempt in range(MAX_RETRIES + 1):
        answer = summarizer_agent(plan, findings)
        evaluation = judge_agent(query, answer, plan, findings)

        if evaluation.passed:
            break
    else:
        # Return best-effort answer even if judge didn't pass
        pass

    return {
        "plan": plan.model_dump(),
        "findings": [f.model_dump() for f in findings],
        "response": answer,
        "evaluation": evaluation.model_dump(),
        "status": "completed" if evaluation.passed else "completed_with_caveats",
    }
