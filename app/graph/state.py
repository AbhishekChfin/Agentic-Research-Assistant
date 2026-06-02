from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    session_id: str
    user_id: str
    query: str
    normalized_query: str
    prior_memories: list[dict[str, Any]]
    plan: dict[str, Any]
    subtasks: list[dict[str, Any]]
    completed_subtasks: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    synthesis: dict[str, Any]
    final_response: str
    citations: list[dict[str, Any]]
    uncertainties: list[str]
    evaluation: dict[str, Any]
    memory_entry: dict[str, Any]
    status: str
    errors: list[dict[str, Any]]