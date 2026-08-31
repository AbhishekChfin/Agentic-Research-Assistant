from pathlib import Path

from app.core.schemas import EvidenceItem, FinalResponse, ResearchPlan
from app.models.answer_llm import get_answer_llm


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summarizer.md"


def synthesize(plan: ResearchPlan, findings: list[EvidenceItem], llm=None) -> FinalResponse:
    answer_llm = llm or get_answer_llm()
    prompt = _build_prompt(plan, findings)
    response = answer_llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    return FinalResponse(
        plan=plan,
        findings=findings,
        response=content,
        status="completed",
    )


def _build_prompt(plan: ResearchPlan, findings: list[EvidenceItem]) -> str:
    prompt_template = PROMPT_PATH.read_text()

    return prompt_template.format(
        query=plan.query,
        plan=plan.model_dump_json(indent=2),
        evidence=_format_evidence(findings),
    )


def _format_evidence(findings: list[EvidenceItem]) -> str:
    evidence = "\n".join(
        (
            f"- Claim: {finding.claim}\n"
            f"  Source: {finding.source_title}\n"
            f"  URL: {finding.source_url}\n"
            f"  Confidence: {finding.confidence}"
        )
        for finding in findings
    )

    return evidence or "No evidence was collected."

