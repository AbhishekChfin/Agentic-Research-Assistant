import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.schemas import EvidenceItem, EvaluationResult, ResearchPlan

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rubrics.md"


class JudgeAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )

    def evaluate_final_answer(
        self,
        query: str,
        answer: str,
        plan: ResearchPlan,
        findings: list[EvidenceItem],
    ) -> EvaluationResult:
        prompt = self._build_prompt(query, answer, plan, findings)
        response = self.llm.with_structured_output(EvaluationResult).invoke(prompt)

        if not isinstance(response, EvaluationResult):
            raise TypeError(
                f"Expected EvaluationResult, got {type(response).__name__}"
            )
        return response

    @staticmethod
    def _build_prompt(
        query: str,
        answer: str,
        plan: ResearchPlan,
        findings: list[EvidenceItem],
    ) -> str:
        prompt_template = PROMPT_PATH.read_text()

        return prompt_template.format(
            query=query,
            answer=answer,
            plan=plan.model_dump_json(indent=2),
            evidence=JudgeAgent._format_evidence(findings),
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
