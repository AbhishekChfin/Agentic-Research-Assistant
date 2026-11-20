"""
agents.py — Specialized agent roles: researcher, summarizer (writer), and judge (editor).

Each agent is a pure function that takes structured inputs and returns
structured outputs — no hidden state, no singletons.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# ── Prompts ─────────────────────────────────────────────────────────

SUMMARIZER_PROMPT = """You are the final answer writer for a research assistant.

Answer the user's query using only the provided evidence.
Do not invent sources or unsupported claims.

User query:
{query}

Research plan:
{plan}

Evidence:
{evidence}

Write a clear, well-structured final answer."""

JUDGE_PROMPT = """You are a strict judge of research quality.

Evaluate whether the answer is relevant to the user's query and whether
the supporting evidence is strong enough to proceed.

User query:
{query}

Agent Summary:
{answer}

Research plan:
{plan}

Evidence collected:
{evidence}

Rules:
1. passed must be true only if the evidence is complete, credible, and relevant.
2. score should be between 0 and 1.
3. summary should be 1-3 sentences.
4. issues should list concrete weaknesses or be empty.
5. missing_evidence should list missing fact areas if any.
6. confidence should be between 0 and 1.
7. notes is optional.
8. If the answer misses an intent, reduce passed and score sharply."""


# ── Helpers ─────────────────────────────────────────────────────────

def _get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )


def _format_evidence(findings) -> str:
    if not findings:
        return "No evidence was collected."
    return "\n".join(
        f"- Claim: {f.claim}\n"
        f"  Source: {f.source_title}\n"
        f"  URL: {f.source_url}\n"
        f"  Confidence: {f.confidence}"
        for f in findings
    )


# ── Researcher agent ───────────────────────────────────────────────

def researcher_agent(plan) -> list:
    """
    Execute each subtask in the plan via web search or RAG,
    and return a list of EvidenceItem objects.
    """
    from src.planning_agent import EvidenceItem
    from src.research_tools import web_search, rag_search

    findings: list = []
    for subtask in plan.subtasks:
        if subtask.mode == "web":
            results = web_search(subtask.task)
        elif subtask.mode == "rag":
            results = rag_search(subtask.task)
        else:
            continue

        for result in results:
            findings.append(
                EvidenceItem(
                    subtask_id=subtask.id,
                    claim=result.claim,
                    source_title=result.source_title,
                    source_url=result.source_url,
                    confidence=result.confidence,
                )
            )
    return findings


# ── Summarizer / Writer agent ──────────────────────────────────────

def summarizer_agent(plan, findings) -> str:
    """Synthesize evidence into a final answer string."""
    llm = _get_llm()
    prompt = SUMMARIZER_PROMPT.format(
        query=plan.query,
        plan=plan.model_dump_json(indent=2),
        evidence=_format_evidence(findings),
    )
    response = llm.invoke(prompt)
    content = response.content
    return content if isinstance(content, str) else str(content)


# ── Judge / Editor agent ───────────────────────────────────────────

def judge_agent(query: str, answer: str, plan, findings):
    """Evaluate the final answer against plan and evidence."""
    from src.planning_agent import EvaluationResult

    llm = _get_llm()
    prompt = JUDGE_PROMPT.format(
        query=query,
        answer=answer,
        plan=plan.model_dump_json(indent=2),
        evidence=_format_evidence(findings),
    )
    structured_llm = llm.with_structured_output(EvaluationResult)
    response = structured_llm.invoke(prompt)

    if not isinstance(response, EvaluationResult):
        raise TypeError(f"Expected EvaluationResult, got {type(response).__name__}")
    return response
