from typing import Any, Callable, Iterable

from app.models import embedding_model


class BGEEmbedder:
    """
    Args:
        emb_dim: Expected embedding vector size for BGE base models.
        emb_model: Optional embedding model instance or factory function.
        batch_size: Number of chunks to embed per model call.

    Returns:
        BGEEmbedder instance that can convert chunk records into embedded records.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        emb_model: Any | Callable[[], Any] = embedding_model.get_embedding_model,
        batch_size: int = 32,
    ):
        if emb_dim <= 0:
            raise ValueError("emb_dim must be greater than 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.emb_dim = emb_dim
        self.emb_model = emb_model() if callable(emb_model) else emb_model
        self.batch_size = batch_size
        self.model_name = embedding_model.model_name

    def create_embeddings(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        embedded_chunks = []

        for batch in self._batch_chunks(chunks):
            texts = [self._get_chunk_text(chunk) for chunk in batch]
            vectors = self.emb_model.encode(
                texts,
                normalize_embeddings=True,
            )

            for chunk, vector in zip(batch, vectors):
                embedding = self._to_list(vector)
                self._validate_embedding(embedding, chunk)

                embedded_chunk = dict(chunk)
                embedded_chunk.update(
                    {
                        "embedding": embedding,
                        "embedding_model": self.model_name,
                        "embedding_dim": len(embedding),
                    }
                )
                embedded_chunks.append(embedded_chunk)

        return embedded_chunks

    def _batch_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> Iterable[list[dict[str, Any]]]:
        for start in range(0, len(chunks), self.batch_size):
            yield chunks[start : start + self.batch_size]

    def _get_chunk_text(self, chunk: dict[str, Any]) -> str:
        text = chunk.get("text", "")
        if not isinstance(text, str) or not text.strip():
            chunk_id = chunk.get("chunk_id", "unknown")
            raise ValueError(f"Chunk {chunk_id} has empty text")

        return text.strip()

    def _to_list(self, vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        return [float(value) for value in vector]

    def _validate_embedding(
        self,
        embedding: list[float],
        chunk: dict[str, Any],
    ) -> None:
        if len(embedding) != self.emb_dim:
            chunk_id = chunk.get("chunk_id", "unknown")
            raise ValueError(
                f"Chunk {chunk_id} produced embedding dim {len(embedding)}, "
                f"expected {self.emb_dim}"
            )

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        
        vector = self.emb_model.encode(
            query.strip(),
            normalize_embeddings=True,
        )

        embedding = self._to_list(vector)
        self._validate_embedding(embedding, {"chunk_id": "query"})

        return embedding