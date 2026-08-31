import re
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer


def estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def clean_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = clean_chunk_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


class BaseChunker:
    chunk_type = "base"

    def chunk(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunked_records = []

        for record_index, record in enumerate(records, start=1):
            text = clean_chunk_text(record.get("text", ""))
            if not text:
                continue

            chunk_texts = self.split_text(text)

            for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
                chunk_text = clean_chunk_text(chunk_text)
                if not chunk_text:
                    continue

                chunked_records.append(
                    self._build_chunk_record(
                        record=record,
                        record_index=record_index,
                        chunk_index=chunk_index,
                        chunk_text=chunk_text,
                    )
                )

        self._add_neighbor_links(chunked_records)
        return chunked_records

    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError

    def _build_chunk_record(
        self,
        record: dict[str, Any],
        record_index: int,
        chunk_index: int,
        chunk_text: str,
    ) -> dict[str, Any]:
        source_stem = Path(record.get("source_title", "source")).stem
        chunk_id = f"{source_stem}_{self.chunk_type}_r{record_index}_c{chunk_index}"

        chunked_record = dict(record)
        chunked_record.update(
            {
                "text": chunk_text,
                "chunk_id": chunk_id,
                "parent_record_id": record.get("record_id", record_index),
                "chunk_index": chunk_index,
                "chunk_size": len(chunk_text),
                "token_count": estimate_token_count(chunk_text),
                "chunk_type": self.chunk_type,
            }
        )
        chunked_record.update(self._extra_metadata(chunk_text))

        return chunked_record

    def _extra_metadata(self, chunk_text: str) -> dict[str, Any]:
        return {}

    def _add_neighbor_links(self, chunks: list[dict[str, Any]]) -> None:
        for index, chunk in enumerate(chunks):
            chunk["prev_chunk_id"] = chunks[index - 1]["chunk_id"] if index > 0 else None
            chunk["next_chunk_id"] = (
                chunks[index + 1]["chunk_id"]
                if index < len(chunks) - 1
                else None
            )


class FixedSizeChunker(BaseChunker):
    chunk_type = "fixed"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])

            if end == len(text):
                break

            start = end - self.chunk_overlap

        return chunks


class TokenChunker(BaseChunker):
    chunk_type = "token"

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer

    def split_text(self, text: str) -> list[str]:
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if not token_ids:
            return []

        chunks = []
        step_size = self.chunk_size - self.chunk_overlap

        for start in range(0, len(token_ids), step_size):
            end = min(start + self.chunk_size, len(token_ids))
            chunk_token_ids = token_ids[start:end]
            chunk_text = self.tokenizer.decode(
                chunk_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            chunks.append(chunk_text)

            if end == len(token_ids):
                break

        return chunks

    def _extra_metadata(self, chunk_text: str) -> dict[str, Any]:
        token_count = len(
            self.tokenizer.encode(
                chunk_text,
                add_special_tokens=False,
            )
        )

        return {
            "token_count": token_count,
            "tokenizer_model": self.model_name,
            "token_window_size": self.chunk_size,
            "token_overlap": self.chunk_overlap,
        }


class SentenceChunker(BaseChunker):
    chunk_type = "sentence"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_chunk and current_size + sentence_size > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = self._get_overlap_sentences(current_chunk)
                current_size = sum(len(item) for item in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        overlap_sentences = []
        overlap_size = 0

        for sentence in reversed(sentences):
            sentence_size = len(sentence)
            if overlap_size + sentence_size <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_size += sentence_size
            else:
                break

        return overlap_sentences


class AdaptiveChunker(BaseChunker):
    chunk_type = "adaptive"

    def __init__(
        self,
        min_chunk_size: int = 600,
        max_chunk_size: int = 1000,
        min_chunk_overlap: int = 30,
        max_chunk_overlap: int = 150,
        complexity_measure: str = "combined",
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_overlap = min_chunk_overlap
        self.max_chunk_overlap = max_chunk_overlap
        self.complexity_measure = complexity_measure

    def analyze_complexity(self, text: str) -> float:
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        lexical_density = len(set(words)) / len(words)
        lexical_density = min(1.0, lexical_density / 0.8)

        sentences = split_sentences(text)
        if sentences:
            avg_sentence_length = sum(len(sentence) for sentence in sentences) / len(sentences)
            sentence_complexity = min(1.0, avg_sentence_length / 200)
        else:
            sentence_complexity = 0.0

        if self.complexity_measure == "lexical_density":
            return lexical_density
        if self.complexity_measure == "sentence_length":
            return sentence_complexity

        return (lexical_density + sentence_complexity) / 2

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        current_complexity = 0.5

        for sentence in sentences:
            sentence_size = len(sentence)
            sentence_complexity = self.analyze_complexity(sentence)

            if current_chunk:
                current_complexity = (current_complexity + sentence_complexity) / 2
            else:
                current_complexity = sentence_complexity

            target_size = self.max_chunk_size - (
                current_complexity * (self.max_chunk_size - self.min_chunk_size)
            )
            target_overlap = self.min_chunk_overlap + (
                current_complexity * (self.max_chunk_overlap - self.min_chunk_overlap)
            )

            if current_chunk and current_size + sentence_size > target_size:
                chunks.append(" ".join(current_chunk))

                overlap_chunk = []
                overlap_size = 0

                for previous_sentence in reversed(current_chunk):
                    previous_size = len(previous_sentence)
                    if overlap_size + previous_size <= target_overlap:
                        overlap_chunk.insert(0, previous_sentence)
                        overlap_size += previous_size
                    else:
                        break

                current_chunk = overlap_chunk + [sentence]
                current_size = sum(len(item) for item in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _extra_metadata(self, chunk_text: str) -> dict[str, Any]:
        return {"text_complexity": round(self.analyze_complexity(chunk_text), 3)}


class RecursiveChunker(BaseChunker):
    chunk_type = "recursive"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        pieces = self._split_recursively(text, self.separators)
        return self._merge_pieces(pieces)

    def _split_recursively(self, text: str, separators: list[str]) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return [
                text[index : index + self.chunk_size]
                for index in range(0, len(text), self.chunk_size)
            ]

        parts = text.split(separator)
        if len(parts) == 1:
            return self._split_recursively(text, remaining_separators)

        pieces = []
        for index, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            if separator == ". " and index < len(parts) - 1:
                part = f"{part}."

            if len(part) > self.chunk_size:
                pieces.extend(self._split_recursively(part, remaining_separators))
            else:
                pieces.append(part)

        return pieces

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        chunks = []
        current = ""

        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece

            if current and len(candidate) > self.chunk_size:
                chunks.append(current)
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = f"{overlap} {piece}".strip() if overlap else piece
            else:
                current = candidate

        if current:
            chunks.append(current)

        return chunks


class SemanticChunker(BaseChunker):
    chunk_type = "semantic"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_chunk and current_size + sentence_size > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = self._get_overlap_sentences(current_chunk)
                current_size = sum(len(item) for item in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        overlap_sentences = []
        overlap_size = 0

        for sentence in reversed(sentences):
            sentence_size = len(sentence)
            if overlap_size + sentence_size <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_size += sentence_size
            else:
                break

        return overlap_sentences


class EmbeddingSemanticChunker(BaseChunker):
    chunk_type = "embedding_semantic"

    def __init__(
        self,
        chunk_size: int = 1000,
        similarity_threshold: float = 0.8,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True,
        )

        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])

        for index in range(1, len(sentences)):
            sentence = sentences[index]
            sentence_size = len(sentence)
            similarity = self._cosine_similarity(
                embeddings[index - 1],
                embeddings[index],
            )

            should_merge = (
                similarity >= self.similarity_threshold
                and current_size + sentence_size <= self.chunk_size
            )

            if should_merge:
                current_chunk.append(sentence)
                current_size += sentence_size
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _cosine_similarity(self, left_embedding: Any, right_embedding: Any) -> float:
        return float(left_embedding @ right_embedding)

    def _extra_metadata(self, chunk_text: str) -> dict[str, Any]:
        return {
            "embedding_model": self.model_name,
            "similarity_threshold": self.similarity_threshold,
        }


def perform_fixed_size_chunking(
    records: list[dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    
    return FixedSizeChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).chunk(records)


def perform_adaptive_chunking(
    records: list[dict[str, Any]],
    min_size: int = 300,
    max_size: int = 1000,
    min_overlap: int = 30,
    max_overlap: int = 150,
    complexity_measure: str = "combined",
) -> list[dict[str, Any]]:
    return AdaptiveChunker(
        min_chunk_size=min_size,
        max_chunk_size=max_size,
        min_chunk_overlap=min_overlap,
        max_chunk_overlap=max_overlap,
        complexity_measure=complexity_measure,
    ).chunk(records)


def perform_recursive_chunking(
    records: list[dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    return RecursiveChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).chunk(records)


def perform_semantic_chunking(
    records: list[dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    return SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).chunk(records)


def perform_token_chunking(
    records: list[dict[str, Any]],
    chunk_size: int = 256,
    chunk_overlap: int = 64,
) -> list[dict[str, Any]]:
    return TokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).chunk(records)


def perform_sentence_chunking(
    records: list[dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    return SentenceChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).chunk(records)


def perform_embedding_semantic_chunking(
    records: list[dict[str, Any]],
    chunk_size: int = 1000,
    similarity_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    return EmbeddingSemanticChunker(
        chunk_size=chunk_size,
        similarity_threshold=similarity_threshold,
    ).chunk(records)


    