import json
from pathlib import Path
from typing import Any

from app.retrieval.chunker import (
    AdaptiveChunker,
    EmbeddingSemanticChunker,
    SemanticChunker,
    SentenceChunker,
)
from app.retrieval.data_extractor import PDFExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "docs" / "financial_machine_learning.pdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"


def save_chunks_to_jsonl(
    chunks: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def save_selected_chunking_strategies(
    records: list[dict[str, Any]] | None = None,
    pdf_path: Path = DEFAULT_PDF_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, int]:
    if records is None:
        records = PDFExtractor(str(pdf_path)).extract()

    chunking_strategies = {
        "financial_machine_learning_sentence_700_overlap_120.jsonl": SentenceChunker(
            chunk_size=700,
            chunk_overlap=120,
        ),
        "financial_machine_learning_semantic_700_overlap_120.jsonl": SemanticChunker(
            chunk_size=700,
            chunk_overlap=120,
        ),
        "financial_machine_learning_embedding_semantic_700_threshold_0_5.jsonl": (
            EmbeddingSemanticChunker(
                chunk_size=700,
                similarity_threshold=0.5,
            )
        ),
        "financial_machine_learning_adaptive_500_800_overlap_80_120.jsonl": AdaptiveChunker(
            min_chunk_size=500,
            max_chunk_size=800,
            min_chunk_overlap=80,
            max_chunk_overlap=120,
        ),
    }

    saved_counts = {}

    for filename, chunker in chunking_strategies.items():
        chunks = chunker.chunk(records)
        output_path = output_dir / filename

        save_chunks_to_jsonl(chunks, output_path)
        saved_counts[output_path.as_posix()] = len(chunks)

    return saved_counts


if __name__ == "__main__":
    saved_counts = save_selected_chunking_strategies()

    for output_path, chunk_count in saved_counts.items():
        print(f"Saved {chunk_count} chunks to {output_path}")
