import os
from typing import Any

import requests
from dotenv import load_dotenv

from app.core.schemas import EvidenceItem


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def web_search(query: str, max_results: int = 3) -> list[EvidenceItem]:
    load_dotenv()

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set")

    response = requests.post(
        TAVILY_SEARCH_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("results", [])

    return [_result_to_evidence(result) for result in results]


def _result_to_evidence(result: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        claim=result.get("content") or "No content returned.",
        source_title=result.get("title") or "Untitled web source",
        source_url=result.get("url") or "",
        confidence=result.get("score"),
    )

# No need to normalize score in tavily, its already on a scale of 1
# def _normalize_confidence(score: Any) -> float:
#     if isinstance(score, (int, float)):
#         return max(0.0, min(float(score), 1.0))

#     return 0.7

# if __name__=="__main__":
#     # Example usage
#     query = "Recent research papers on agent memory systems"
#     results = web_search(query)
#     for item in results:
#         print(f"Claim: {item.claim}")
#         print(f"Source: {item.source_title} ({item.source_url})")
#         print(f"Confidence: {item.confidence}")
#         print("-" * 40)