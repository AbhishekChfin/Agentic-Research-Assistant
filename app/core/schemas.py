from pydantic import BaseModel
from typing import List, Optional, Literal


class Subtask(BaseModel):
    id: str
    task: str
    mode: Literal["web", "rag"]
    success_criteria: Optional[str] = None


class ResearchPlan(BaseModel):
    query: str
    subtasks: List[Subtask]


class EvidenceItem(BaseModel):
    claim: str
    source_title: str
    source_url: str
    confidence: float

class FinalResponse(BaseModel):
    plan: ResearchPlan
    findings: List[EvidenceItem]
    response: str
    status: str


class MemoryEntry(BaseModel):
    query: str
    findings: List[EvidenceItem]
    summary: str
    tags: List[str]


class EvaluationResult(BaseModel):
    pass