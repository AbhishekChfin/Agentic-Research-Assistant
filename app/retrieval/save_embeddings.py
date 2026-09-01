from pathlib import Path
import json
from app.retrieval.embedder import BGEEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_chunks_from_jsonl(input_path: Path) -> list[dict]:
    """Read JSONL file, return list of dicts"""
    chunks = []
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def save_embeddings_to_jsonl(embedded_chunks: list[dict], output_path: Path) -> None:
    """Write list of dicts to JSONL file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in embedded_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
def embed_chunks_file(
    input_path: Path,
    output_path: Path,
    batch_size: int = 32
) -> int:
    """Load chunks → embed → save, return count"""
    chunks = load_chunks_from_jsonl(input_path)
    embedder = BGEEmbedder(batch_size=batch_size)
    embedded_chunks = embedder.create_embeddings(chunks)
    save_embeddings_to_jsonl(embedded_chunks, output_path)
    return len(embedded_chunks)

# if __name__ == "__main__":
#     input_file = PROJECT_ROOT / "data/processed/chunks/financial_machine_learning_sentence_700_overlap_120.jsonl"
#     output_file = PROJECT_ROOT / "data/processed/embeddings/financial_machine_learning_sentence_700_overlap_120_embedded.jsonl"
    
#     count = embed_chunks_file(input_file, output_file)
#     print(f"Embedded and saved {count} chunks to {output_file}")