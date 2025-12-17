import pytest
from pathlib import Path
from types import SimpleNamespace

from app.services.vector_store import (
    VectorStore,
    load_vector_store,
    save_vector_store,
)


# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture
def settings(tmp_path: Path):
    return SimpleNamespace(vectorstore_path=str(tmp_path))


@pytest.fixture
def embeddings_3d():
    # Orthonormal vectors (cosine similarity is exact)
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


@pytest.fixture
def metadata_3d():
    return [
        {"text": "doc1", "doc_id": "1", "chunk_index": 0},
        {"text": "doc2", "doc_id": "2", "chunk_index": 0},
        {"text": "doc3", "doc_id": "3", "chunk_index": 0},
    ]


# ----------------------------
# VectorStore unit tests
# ----------------------------


def test_create_index():
    store = VectorStore()
    store.create_index(3)

    assert store.index is not None
    assert store.dimension == 3
    assert store.ntotal == 0
    assert store.metadata == []


def test_add_embeddings_success(embeddings_3d, metadata_3d):
    store = VectorStore()
    store.create_index(3)

    store.add_embeddings(embeddings_3d, metadata_3d)

    assert store.ntotal == 3
    assert len(store.metadata) == 3


def test_add_embeddings_length_mismatch(embeddings_3d, metadata_3d):
    store = VectorStore()
    store.create_index(3)

    with pytest.raises(ValueError, match="length mismatch"):
        store.add_embeddings(embeddings_3d, metadata_3d[:2])


def test_add_embeddings_dimension_mismatch(metadata_3d):
    store = VectorStore()
    store.create_index(4)

    embeddings = [[1.0, 0.0, 0.0]] * 3

    with pytest.raises(ValueError, match="dimension"):
        store.add_embeddings(embeddings, metadata_3d)


def test_add_embeddings_invalid_metadata(embeddings_3d):
    store = VectorStore()
    store.create_index(3)

    bad_metadata = [{"text": "x", "doc_id": "1"}] * 3

    with pytest.raises(ValueError, match="required keys"):
        store.add_embeddings(embeddings_3d, bad_metadata)


def test_search_returns_sorted_results(embeddings_3d, metadata_3d):
    store = VectorStore()
    store.create_index(3)
    store.add_embeddings(embeddings_3d, metadata_3d)

    query = [1.0, 0.0, 0.0]
    results = store.search(query, top_k=3)

    assert len(results) == 3
    assert results[0]["text"] == "doc1"
    assert results[0]["score"] > results[1]["score"]


def test_search_dimension_mismatch(embeddings_3d, metadata_3d):
    store = VectorStore()
    store.create_index(3)
    store.add_embeddings(embeddings_3d, metadata_3d)

    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0, 0.0], top_k=1)


def test_search_handles_top_k_greater_than_ntotal(embeddings_3d, metadata_3d):
    store = VectorStore()
    store.create_index(3)
    store.add_embeddings(embeddings_3d, metadata_3d)

    results = store.search([1.0, 0.0, 0.0], top_k=10)
    assert len(results) == 3


def test_search_empty_store_returns_empty():
    store = VectorStore()
    store.create_index(3)

    assert store.search([1.0, 0.0, 0.0]) == []


# ----------------------------
# Persistence tests
# ----------------------------


def test_save_and_load_roundtrip(settings, embeddings_3d, metadata_3d):
    store = VectorStore()
    store.create_index(3)
    store.add_embeddings(embeddings_3d, metadata_3d)

    save_vector_store(store, settings)

    loaded_store, loaded_metadata = load_vector_store(settings)

    assert loaded_store.ntotal == 3
    assert loaded_store.dimension == 3
    assert loaded_metadata == metadata_3d

    results = loaded_store.search([0.0, 1.0, 0.0], top_k=1)
    assert results[0]["text"] == "doc2"


def test_load_corrupted_index_recovers(settings):
    path = Path(settings.vectorstore_path)
    path.mkdir(parents=True, exist_ok=True)

    # Create corrupted files
    (path / "index.faiss").write_text("corrupted")
    (path / "metadata.pkl").write_bytes(b"not-a-pickle")

    store, metadata = load_vector_store(settings)

    assert store.index is not None
    assert store.ntotal == 0
    assert store.dimension == 384
    assert metadata == []


def test_save_skips_empty_index(settings):
    store = VectorStore()
    store.create_index(3)

    save_vector_store(store, settings)

    assert not (Path(settings.vectorstore_path) / "index.faiss").exists()
    assert not (Path(settings.vectorstore_path) / "metadata.pkl").exists()
