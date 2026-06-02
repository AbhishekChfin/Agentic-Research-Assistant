import pytest
import json
from pathlib import Path
from app.retrieval.embedder import BGEEmbedder
from app.retrieval.save_embeddings import (
    load_chunks_from_jsonl,
    save_embeddings_to_jsonl,
    embed_chunks_file,
)

@pytest.fixture
def sample_chunks():
    return [
        {"text": "Sample chunk 1", "chunk_id": "c1"},
        {"text": "Sample chunk 2", "chunk_id": "c2"},
    ]

@pytest.fixture
def sample_chunks_with_metadata():
    return [
        {
            "text": "Sample chunk 1",
            "chunk_id": "c1",
            "source_title": "test.pdf",
            "source_url": "data/test.pdf",
            "pdf_page": 1,
        },
        {
            "text": "Sample chunk 2",
            "chunk_id": "c2",
            "source_title": "test.pdf",
            "source_url": "data/test.pdf",
            "pdf_page": 2,
        },
    ]

@pytest.fixture
def large_sample_chunks():
    """Create 100 chunks to test batch processing"""
    return [
        {"text": f"Sample chunk {i}", "chunk_id": f"c{i}"}
        for i in range(100)
    ]

@pytest.fixture
def temp_jsonl(tmp_path):
    """Create a temporary JSONL file for testing"""
    chunks = [
        {"text": "Chunk 1", "chunk_id": "c1", "pdf_page": 1},
        {"text": "Chunk 2", "chunk_id": "c2", "pdf_page": 2},
    ]
    jsonl_path = tmp_path / "test_chunks.jsonl"
    with jsonl_path.open("w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    return jsonl_path, chunks

# ============ BGEEmbedder Tests ============

def test_embedder_initializes():
    embedder = BGEEmbedder()
    assert embedder.emb_dim == 768
    assert embedder.batch_size == 32

def test_create_embeddings_adds_fields(sample_chunks):
    embedder = BGEEmbedder()
    embedded = embedder.create_embeddings(sample_chunks)
    
    assert len(embedded) == 2
    assert "embedding" in embedded[0]
    assert len(embedded[0]["embedding"]) == 768
    assert embedded[0]["embedding_model"] == "BAAI/bge-base-en-v1.5"
    assert embedded[0]["embedding_dim"] == 768

def test_empty_text_raises_error():
    embedder = BGEEmbedder()
    bad_chunks = [{"text": "", "chunk_id": "bad"}]
    
    with pytest.raises(ValueError):
        embedder.create_embeddings(bad_chunks)

def test_embedding_is_normalized(sample_chunks):
    """Check that embeddings are normalized (L2 norm ≈ 1)"""
    embedder = BGEEmbedder()
    embedded = embedder.create_embeddings(sample_chunks)
    
    for chunk in embedded:
        embedding = chunk["embedding"]
        # L2 norm should be close to 1.0 for normalized embeddings
        l2_norm = sum(x**2 for x in embedding) ** 0.5
        assert 0.99 < l2_norm < 1.01

def test_metadata_preserved(sample_chunks_with_metadata):
    """Check that original metadata is preserved in embedded chunks"""
    embedder = BGEEmbedder()
    embedded = embedder.create_embeddings(sample_chunks_with_metadata)
    
    # Check first chunk
    assert embedded[0]["chunk_id"] == "c1"
    assert embedded[0]["source_title"] == "test.pdf"
    assert embedded[0]["source_url"] == "data/test.pdf"
    assert embedded[0]["pdf_page"] == 1

def test_batch_processing_works(large_sample_chunks):
    """Test that embedding handles batches correctly (>32 chunks with batch_size=32)"""
    embedder = BGEEmbedder(batch_size=32)
    embedded = embedder.create_embeddings(large_sample_chunks)
    
    assert len(embedded) == 100
    for chunk in embedded:
        assert len(chunk["embedding"]) == 768

def test_custom_batch_size():
    embedder = BGEEmbedder(batch_size=16)
    assert embedder.batch_size == 16

def test_custom_emb_dim_validation():
    with pytest.raises(ValueError):
        BGEEmbedder(emb_dim=0)
    
    with pytest.raises(ValueError):
        BGEEmbedder(batch_size=0)

# ============ File I/O Tests ============

def test_load_chunks_from_jsonl(temp_jsonl):
    jsonl_path, expected_chunks = temp_jsonl
    loaded = load_chunks_from_jsonl(jsonl_path)
    
    assert len(loaded) == 2
    assert loaded[0]["chunk_id"] == "c1"
    assert loaded[1]["chunk_id"] == "c2"

def test_save_embeddings_to_jsonl(tmp_path, sample_chunks):
    embedder = BGEEmbedder()
    embedded = embedder.create_embeddings(sample_chunks)
    
    output_path = tmp_path / "test_embedded.jsonl"
    save_embeddings_to_jsonl(embedded, output_path)
    
    assert output_path.exists()
    
    # Verify content
    loaded = load_chunks_from_jsonl(output_path)
    assert len(loaded) == 2
    assert "embedding" in loaded[0]
    assert len(loaded[0]["embedding"]) == 768

def test_save_creates_parent_directories(tmp_path):
    """Verify that save_embeddings_to_jsonl creates parent directories"""
    embedder = BGEEmbedder()
    chunks = [{"text": "Test", "chunk_id": "c1"}]
    embedded = embedder.create_embeddings(chunks)
    
    nested_path = tmp_path / "dir1" / "dir2" / "embeddings.jsonl"
    save_embeddings_to_jsonl(embedded, nested_path)
    
    assert nested_path.exists()
    assert nested_path.parent.exists()

# ============ Round-Trip Tests ============

def test_round_trip_load_embed_save(tmp_path):
    """Test the full pipeline: load chunks → embed → save"""
    # Create test chunks
    chunks = [
        {"text": "Test chunk 1", "chunk_id": "c1", "source_url": "test.pdf"},
        {"text": "Test chunk 2", "chunk_id": "c2", "source_url": "test.pdf"},
    ]
    
    # Save chunks to JSONL
    input_path = tmp_path / "chunks.jsonl"
    with input_path.open("w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    
    # Embed and save
    output_path = tmp_path / "embeddings.jsonl"
    count = embed_chunks_file(input_path, output_path)
    
    assert count == 2
    assert output_path.exists()
    
    # Load and verify
    embedded = load_chunks_from_jsonl(output_path)
    assert len(embedded) == 2
    assert all("embedding" in chunk for chunk in embedded)
    assert all(len(chunk["embedding"]) == 768 for chunk in embedded)
    assert embedded[0]["chunk_id"] == "c1"
    assert embedded[0]["source_url"] == "test.pdf"