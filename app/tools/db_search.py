"""
No layoutparser
No detectron2
Still extract:
- page text
- tables
- table id/title from nearby text
- table rows in sentence format
- relative source_url
"""
import re
from pathlib import Path
from typing import Any
import pdfplumber
from app.core.schemas import EvidenceItem
from app.retrieval.embedder import BGEEmbedder
from app.retrieval.vector_index import VectorIndex

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    /"data"
    /"processed"
    /"embeddings"
    /"financial_machine_learning_sentence_700_overlap_120_embedded.jsonl"
)


class RAGSearch:
    """
    Args:
        embeddings_path: JSONL file containing embedded chunk records.
        top_k: Default number of matching chunks to return.

    Returns:
        RAGSearch instance that lazily builds and reuses a vector index.
    """

    def __init__(
        self,
        embeddings_path: Path = EMBEDDINGS_PATH,
        top_k: int = 10, 
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self.embeddings_path = embeddings_path
        self.top_k = top_k
        self.embedder = BGEEmbedder()
        self._vector_index: VectorIndex | None = None

    def _get_vector_index(self) -> VectorIndex:
        if self._vector_index is None:
            vector_index = VectorIndex(embedder=self.embedder)
            records = vector_index.load_embeddings(self.embeddings_path)
            vector_index.build(records)
            self._vector_index = vector_index

        return self._vector_index

    def search(self, query: str, top_k: int | None = None) -> list[EvidenceItem]:
        result_limit = self.top_k if top_k is None else top_k
        if result_limit <= 0:
            raise ValueError("top_k must be greater than 0")

        results = self._get_vector_index().search(query, top_k=result_limit)

        return [
            EvidenceItem(
                claim=result["text"],
                source_title=result["source_title"],
                source_url=result["source_url"],
                confidence=result["similarity_score"],
            )
            for result in results
        ]