"""
research_tools.py — External tool wrappers for the research agents.

Tools:
    web_search   → Tavily API for real-time web results
    rag_search   → Local FAISS vector index over embedded PDF chunks
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import arxiv
import numpy as np
import requests
import wikipedia
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Evidence schema ─────────────────────────────────────────────────

class Evidence(BaseModel):
    claim: str
    source_title: str
    source_url: str
    confidence: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WEB SEARCH  (Tavily)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def web_search(query: str, max_results: int = 3) -> list[Evidence]:
    """Search the web via Tavily and return Evidence objects."""
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

    results = response.json().get("results", [])
    return [
        Evidence(
            claim=r.get("content") or "No content returned.",
            source_title=r.get("title") or "Untitled web source",
            source_url=r.get("url") or "",
            confidence=r.get("score", 0.7),
        )
        for r in results
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RAG SEARCH  (FAISS + BGE embeddings)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "financial_machine_learning_sentence_700_overlap_120_embedded.jsonl"
)

# Lazy singletons
_embedder = None
_vector_index = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _embedder


def _get_vector_index():
    """Build the FAISS index once and reuse it."""
    global _vector_index
    if _vector_index is not None:
        return _vector_index

    import faiss

    records = _load_embeddings(EMBEDDINGS_PATH)
    if not records:
        raise RuntimeError(f"No embeddings found at {EMBEDDINGS_PATH}")

    vectors = np.array([r["embedding"] for r in records], dtype=np.float32)
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    _vector_index = {"index": index, "records": records, "dim": dim}
    return _vector_index


def _load_embeddings(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def rag_search(query: str, top_k: int = 10) -> list[Evidence]:
    """Search the local vector index and return Evidence objects."""
    import faiss

    model = _get_embedder()
    vi = _get_vector_index()

    query_vec = model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)
    faiss.normalize_L2(query_vec)

    scores, positions = vi["index"].search(query_vec, min(top_k, len(vi["records"])))

    results = []
    for score, pos in zip(scores[0], positions[0]):
        rec = vi["records"][pos]
        results.append(
            Evidence(
                claim=rec.get("text", ""),
                source_title=rec.get("source_title", ""),
                source_url=rec.get("source_url", ""),
                confidence=float(score),
            )
        )
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ARXIV SEARCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def arxiv_search(query: str, max_results: int = 3) -> list[Evidence]:
    """Search arXiv for academic papers and return Evidence objects."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for paper in client.results(search):
        # Use title + truncated abstract as the claim
        abstract = paper.summary.replace("\n", " ").strip()
        if len(abstract) > 500:
            abstract = abstract[:497] + "..."

        results.append(
            Evidence(
                claim=f"{paper.title}. {abstract}",
                source_title=paper.title,
                source_url=paper.entry_id,
                confidence=0.85,
            )
        )
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WIKIPEDIA SEARCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def wikipedia_search(query: str, max_results: int = 3) -> list[Evidence]:
    """Search Wikipedia for background knowledge and return Evidence objects."""
    results = []

    try:
        page_titles = wikipedia.search(query, results=max_results)
    except Exception:
        return results

    for title in page_titles:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            summary = page.summary
            if len(summary) > 500:
                summary = summary[:497] + "..."

            results.append(
                Evidence(
                    claim=summary,
                    source_title=page.title,
                    source_url=page.url,
                    confidence=0.75,
                )
            )
        except (wikipedia.DisambiguationError, wikipedia.PageError):
            continue

    return results
