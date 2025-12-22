import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock

from app.api.v1.ingest import router
from app.core.deps import set_embedding_model, set_vector_store
from app.services.embedding_service import EmbeddingModel
from app.services.vector_store import VectorStore


# -------------------------------------------------
# Test App
# -------------------------------------------------


@pytest.fixture()
def test_app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/ingest")
    return app


@pytest.fixture()
def client(test_app):
    return TestClient(test_app)


# -------------------------------------------------
# Mocks
# -------------------------------------------------


@pytest.fixture()
def mock_embedding_model():
    model = Mock(spec=EmbeddingModel)
    model.embed_batch.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    return model


@pytest.fixture()
def mock_vector_store():
    store = Mock(spec=VectorStore)
    store.ntotal = 0
    store.dimension = 3
    store.index = Mock()
    store.index.__class__.__name__ = "IndexFlatL2"

    # IMPORTANT: stub mutating method
    store.add_embeddings.return_value = None

    return store


@pytest.fixture(autouse=True)
def inject_dependencies(mock_embedding_model, mock_vector_store):
    set_embedding_model(mock_embedding_model)
    set_vector_store(mock_vector_store)


# -------------------------------------------------
# Tests
# -------------------------------------------------


def test_ingest_text_success(client, mock_vector_store):
    payload = {
        "text": "This is a test document. " * 20,
    }

    response = client.post("/api/v1/ingest/text", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["doc_id"].startswith("doc-")
    assert data["chunks_added"] > 0
    assert "total_docs" in data

    mock_vector_store.add_embeddings.assert_called_once()


def test_ingest_text_with_explicit_doc_id(client):
    payload = {
        "text": "Another test document. " * 20,
        "doc_id": "my-doc-id",
        "chunk_size": 200,
        "chunk_overlap": 50,
    }

    response = client.post("/api/v1/ingest/text", json=payload)

    assert response.status_code == 200
    assert response.json()["doc_id"] == "my-doc-id"


def test_ingest_text_validation_error(client):
    payload = {
        "text": "",
        "chunk_size": 200,
        "chunk_overlap": 50,
    }

    response = client.post("/api/v1/ingest/text", json=payload)

    assert response.status_code == 422


def test_index_status(client):
    response = client.get("/api/v1/ingest/status")

    assert response.status_code == 200
    data = response.json()

    assert data["embedding_dimension"] == 3
    assert data["index_type"] == "IndexFlatL2"
