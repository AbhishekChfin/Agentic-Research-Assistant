import math
import re
import time
from pathlib import Path

import pytest

from app.retrieval.chunker import (
    AdaptiveChunker,
    BaseChunker,
    EmbeddingSemanticChunker,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
    perform_adaptive_chunking,
    perform_embedding_semantic_chunking,
    perform_fixed_size_chunking,
    perform_sentence_chunking,
    perform_token_chunking,
)
from app.retrieval.rag_pipeline import PDFExtractor


SAMPLE_PDF_PATH = Path(__file__).resolve().parent.parent.parent/'data'/'docs'/'financial_machine_learning.pdf'


@pytest.fixture(scope="module")
def sample_records() -> list[dict]:
    return PDFExtractor(str(SAMPLE_PDF_PATH)).extract()


keywords = [
    "machine learning",
    "financial machine learning",
    "prediction",
    "risk",
    "return",
    "portfolio",
    "regression",
    "training",
]


key_phrases = [
    "financial machine learning",
    "predictive models",
    "training data",
    "out-of-sample test data",
    "linear regression",
]


chunking_strategies = {
    "fixed_700_overlap_120": FixedSizeChunker(chunk_size=700, chunk_overlap=120),
    "token_256_overlap_64": TokenChunker(chunk_size=256, chunk_overlap=64),
    "sentence_700_overlap_120": SentenceChunker(chunk_size=700, chunk_overlap=120),
    "adaptive_500_800": AdaptiveChunker(
        min_chunk_size=500,
        max_chunk_size=800,
        min_chunk_overlap=80,
        max_chunk_overlap=120,
    ),
    "recursive_700_overlap_120": RecursiveChunker(chunk_size=700, chunk_overlap=120),
    "semantic_700_overlap_120": SemanticChunker(chunk_size=700, chunk_overlap=120),
    "embedding_semantic_700": EmbeddingSemanticChunker(
        chunk_size=700,
        similarity_threshold=0.5,
    ),
}


def calculate_keyword_coverage(chunks: list[dict], keywords: list[str]) -> float:
    chunk_texts = [chunk.get("text", "").lower() for chunk in chunks]
    keywords = [keyword.lower() for keyword in keywords]

    found = 0
    for keyword in keywords:
        if any(keyword in text for text in chunk_texts):
            found += 1

    return found / max(1, len(keywords))


def calculate_chunk_coherence(chunks: list[dict]) -> float:
    incomplete_boundaries = 0

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        if text[0].islower() or text[0] in ",;:)]}":
            incomplete_boundaries += 1

        if not re.search(r"[.!?]\s*$", text):
            incomplete_boundaries += 1

    max_boundaries = len(chunks) * 2
    return 1 - (incomplete_boundaries / max(1, max_boundaries))


def calculate_concept_splitting(chunks: list[dict], key_phrases: list[str]) -> float:
    chunk_texts = [chunk.get("text", "").lower() for chunk in chunks]
    split_phrases = 0

    for phrase in key_phrases:
        phrase = phrase.lower()
        words = phrase.split()

        if not words:
            continue

        complete_in_chunk = any(phrase in text for text in chunk_texts)
        if complete_in_chunk or len(words) == 1:
            continue

        parts_split = False

        for index in range(1, len(words)):
            part1 = " ".join(words[:index])
            part2 = " ".join(words[index:])

            part1_chunks = [idx for idx, text in enumerate(chunk_texts) if part1 in text]
            part2_chunks = [idx for idx, text in enumerate(chunk_texts) if part2 in text]

            if part1_chunks and part2_chunks and min(part2_chunks) > max(part1_chunks):
                parts_split = True
                break

        if parts_split:
            split_phrases += 1

    return 1 - (split_phrases / max(1, len(key_phrases)))


def evaluate_chunking_strategies(
    records: list[dict],
    strategies: dict[str, BaseChunker],
    keywords: list[str],
    key_phrases: list[str],
) -> list[dict]:
    results = []

    for name, chunker in strategies.items():
        start_time = time.time()
        chunks = chunker.chunk(records)
        processing_time = time.time() - start_time

        chunk_sizes = [chunk["chunk_size"] for chunk in chunks]
        avg_chunk_size = sum(chunk_sizes) / max(1, len(chunk_sizes))
        chunk_size_std = (
            math.sqrt(sum((size - avg_chunk_size) ** 2 for size in chunk_sizes) / len(chunk_sizes))
            if chunk_sizes
            else 0
        )
        size_consistency = 1 - (chunk_size_std / max(1, avg_chunk_size))

        results.append(
            {
                "strategy": name,
                "processing_time": round(processing_time, 4),
                "keyword_coverage": round(calculate_keyword_coverage(chunks, keywords), 2),
                "chunk_coherence": round(calculate_chunk_coherence(chunks), 2),
                "concept_integrity": round(calculate_concept_splitting(chunks, key_phrases), 2),
                "size_consistency": round(size_consistency, 2),
                "total_chunks": len(chunks),
                "avg_chunk_size": round(avg_chunk_size, 2),
            }
        )

    return results


def plot_results(results: list[dict]) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        "processing_time",
        "keyword_coverage",
        "concept_integrity",
        "chunk_coherence",
        "total_chunks",
        "size_consistency",
    ]
    strategies = [result["strategy"] for result in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for axis, metric in zip(axes, metrics):
        axis.bar(strategies, [result[metric] for result in results])
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def test_all_chunkers_return_valid_records(sample_records):
    for chunker in chunking_strategies.values():
        chunks = chunker.chunk(sample_records)

        assert chunks
        assert all(chunk["text"].strip() for chunk in chunks)

        for chunk in chunks:
            assert chunk["source_title"]
            assert chunk["source_url"]
            assert chunk["page"]
            assert chunk["content_type"]
            assert chunk["chunk_id"]
            assert chunk["chunk_index"] >= 1
            assert chunk["chunk_size"] == len(chunk["text"])
            assert chunk["token_count"] >= 1
            assert "prev_chunk_id" in chunk
            assert "next_chunk_id" in chunk


def test_existing_wrapper_functions_still_work(sample_records):
    fixed_chunks = perform_fixed_size_chunking(sample_records, chunk_size=700, chunk_overlap=120)
    token_chunks = perform_token_chunking(sample_records, chunk_size=256, chunk_overlap=64)
    sentence_chunks = perform_sentence_chunking(sample_records, chunk_size=700, chunk_overlap=120)
    adaptive_chunks = perform_adaptive_chunking(sample_records, min_size=80, max_size=180)
    embedding_semantic_chunks = perform_embedding_semantic_chunking(
        sample_records,
        chunk_size=700,
        similarity_threshold=0.5,
    )

    assert fixed_chunks
    assert token_chunks
    assert sentence_chunks
    assert adaptive_chunks
    assert embedding_semantic_chunks
    assert fixed_chunks[0]["chunk_type"] == "fixed"
    assert token_chunks[0]["chunk_type"] == "token"
    assert sentence_chunks[0]["chunk_type"] == "sentence"
    assert adaptive_chunks[0]["chunk_type"] == "adaptive"
    assert embedding_semantic_chunks[0]["chunk_type"] == "embedding_semantic"


def test_evaluate_chunking_strategies_returns_metrics(sample_records):
    results = evaluate_chunking_strategies(
        records=sample_records,
        strategies=chunking_strategies,
        keywords=keywords,
        key_phrases=key_phrases,
    )

    assert len(results) == len(chunking_strategies)

    for result in results:
        assert result["strategy"]
        assert result["total_chunks"] > 0
        assert 0 <= result["keyword_coverage"] <= 1
        assert 0 <= result["chunk_coherence"] <= 1
        assert 0 <= result["concept_integrity"] <= 1

from tabulate import tabulate
if __name__ == "__main__":
    sample_records = PDFExtractor(str(SAMPLE_PDF_PATH)).extract()

    results = evaluate_chunking_strategies(
        records=sample_records,
        strategies=chunking_strategies,
        keywords=keywords,
        key_phrases=key_phrases,
    )

    print("\nEvaluation results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))

    for name, chunker in chunking_strategies.items():
        chunks = chunker.chunk(sample_records)
        print(f"\n===== {name} =====")
        print(f"Total chunks: {len(chunks)}")

        for chunk in chunks[:2]:
            print("\n--- SAMPLE CHUNK ---")
            print("chunk_id:", chunk["chunk_id"])
            print("page:", chunk.get("page"))
            print("source_url:", chunk.get("source_url"))
            print("chunk_size:", chunk["chunk_size"])
            print("token_count:", chunk["token_count"])
            print("prev_chunk_id:", chunk["prev_chunk_id"])
            print("next_chunk_id:", chunk["next_chunk_id"])
            print(chunk["text"])
