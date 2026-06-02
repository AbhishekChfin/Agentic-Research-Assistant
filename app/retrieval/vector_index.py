from pathlib import Path
import json

import faiss
import numpy as np
from typing import Any
from app.retrieval.embedder import BGEEmbedder


class VectorIndex:
    '''
    Args:
        emb_dim: Expected embedding vector dimension
    
    Returns:
        VectorIndex instance for loading, indexing and search embed chunks
    '''
    def __init__(self, emb_dim: int = 768, embedder: BGEEmbedder | None=None,):
        if emb_dim<=0:
            raise ValueError("Embedding dimension must be greater than 0")
        
        self.emb_dim = emb_dim
        self.index = faiss.IndexFlatIP(emb_dim)
        self.records: list[dict[str, Any]] = []
        self.embedder = embedder or BGEEmbedder(emb_dim=emb_dim)

    @staticmethod
    def load_embeddings(input_path: Path) -> list[dict[str, Any]]:
        records = []

        with input_path.open('r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
        return records
    
    def build(self, records: list[dict[str, Any]]) -> None:

        if not records:
            raise ValueError("Records must not be empty")
        
        vectors = []

        for record in records:
            embedding = record.get("embedding")

            if not isinstance(embedding, list):
                raise ValueError("Every record must contain an embedding list")
            
            
            if len(embedding)!=self.emb_dim:
                raise ValueError(
                    f"Expected embedding dim {self.emb_dim}, got {len(embedding)}"
                )
            vectors.append(embedding)

        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        self.index.add(matrix)
        self.records.extend(records)
    
    def search_by_vector(
        self,
        query_embedding: list[float],
        top_k: int = 10
    ) -> list[dict[str, Any]]:
        
        if not self.records:
            raise ValueError("Index is empty. Build the index before searching.")
        
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        
        query = np.asarray([query_embedding], dtype="float32")

        if query.shape[1] != self.emb_dim:
            raise ValueError(
                f"Expected query dim {self.emb_dim}, got {query.shape[1]}"
            )
        
        faiss.normalize_L2(query)

        scores, positions = self.index.search(
            query, 
            min(top_k, len(self.records)),
        )

        results = []
        for score, position in zip(scores[0], positions[0]):
            record = dict(self.records[position])
            record["similarity_score"] = float(score)
            results.append(record)
        
        return results
    
    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query_embedding = self.embedder.embed_query(query)
        return self.search_by_vector(query_embedding, top_k=top_k)
