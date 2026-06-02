import pytest

from app.retrieval.vector_index import VectorIndex


@pytest.fixture
def sample_records():
    return [
        {"chunk_id": "c1", "text": "machine learning", "embedding": [1.0, 0.0]},
        {"chunk_id": "c2", "text": "portfolio risk", "embedding": [0.0, 1.0]},
    ]


def test_build_adds_records(sample_records):
    index = VectorIndex(emb_dim=2)

    index.build(sample_records)

    assert index.index.ntotal == 2
    assert len(index.records) == 2


def test_search_returns_closest_chunk(sample_records):
    index = VectorIndex(emb_dim=2)
    index.build(sample_records)

    results = index.search_by_vector([0.9, 0.1], top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert "similarity_score" in results[0]


def test_empty_records_raise_error():
    index = VectorIndex(emb_dim=2)

    with pytest.raises(ValueError):
        index.build([])


def test_invalid_query_dimension_raises_error(sample_records):
    index = VectorIndex(emb_dim=2)
    index.build(sample_records)

    with pytest.raises(ValueError):
        index.search_by_vector([1.0, 0.0, 0.0])