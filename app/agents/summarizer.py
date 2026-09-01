from pathlib import Path

from app.core.schemas import EvidenceItem, FinalResponse, ResearchPlan
from app.models.answer_llm import get_answer_llm


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summarizer.md"

class SummariserAgent:

    def __init__(self):
        self._llm = get_answer_llm()

    def synthesize(self, plan: ResearchPlan, findings: list[EvidenceItem]) -> FinalResponse:
        prompt = self._build_prompt(plan, findings)
        response = self._llm.invoke(prompt)
        content = response.content

        if not isinstance(content, str):
            content = str(content)
            
        return FinalResponse(
            plan=plan,
            findings=findings,
            response=content,
            status="completed"
        )
    
    @staticmethod
    def _build_prompt(plan: ResearchPlan, findings: list[EvidenceItem]) -> str:
        prompt_template = PROMPT_PATH.read_text()

        return prompt_template.format(
            query=plan.query,
            plan=plan.model_dump_json(indent=2),
            evidence=SummariserAgent._format_evidence(findings),
        )

    @staticmethod
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

