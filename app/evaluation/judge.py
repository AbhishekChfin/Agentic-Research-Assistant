import json
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.schemas import EvidenceItem, EvaluationResult, ResearchPlan

PROMPT_PATH = Path(__file__).resolve().parent / "judge_prompt.md"


class JudgeAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )

    def evaluate_response(
        self,
        query: str,
        plan: ResearchPlan,
        findings: list[EvidenceItem],
    ) -> EvaluationResult:
        prompt = self._build_prompt(query, plan, findings)
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        payload = self._extract_json(content)
        return EvaluationResult.model_validate(payload)

    def _build_prompt(self, query: str, plan: ResearchPlan, findings: list[EvidenceItem]) -> str:
        schema = EvaluationResult.model_json_schema()
        evidence = "\n".join(
            (
                f"- Claim: {finding.claim}\n"
                f"  Source: {finding.source_title}\n"
                f"  URL: {finding.source_url}\n"
                f"  Confidence: {finding.confidence}"
            )
            for finding in findings
        ) or "No evidence was collected."

        prompt = f"""
You are a strict research quality judge.

Your job is to decide whether the current research output is good enough to proceed to final synthesis.

Task:
- Evaluate the completeness, quality, and evidence coverage of the current findings.
- Consider if the facts answer the user's query, whether critical claims are supported, and whether the evidence is sufficiently strong.
- If the research is weak, identify what is missing and suggest the next action.

User query:
{query}

Research plan:
{plan.model_dump_json(indent=2)}

Evidence collected:
{evidence}

Return valid JSON that matches this schema exactly:
{schema}

Rules:
1. passed must be true only if the evidence is sufficiently complete and credible.
2. score should be a number between 0 and 1.
3. summary should briefly explain the verdict in 1-3 sentences.
4. issues should be a list of concrete weaknesses, or an empty list if none.
5. missing_evidence should be a list of missing fact areas or questions still unanswered.
6. recommended_action should be one of: "continue", "replan", "research_more", "stop_and_summarize".
7. confidence should be a number between 0 and 1.
8. notes is optional and can be null or omitted.
"""
        return prompt

    def _extract_json(self, content: str) -> dict:
        text = content.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").removesuffix("```").strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end < start:
            raise ValueError(f"Judge did not return JSON: {content}")

        return json.loads(text[start : end + 1])
