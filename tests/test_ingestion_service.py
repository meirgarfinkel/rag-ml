from unittest.mock import Mock

from app.services.ingestion_service import (
    process_text,
    chunk_text,
)
from app.services.vector_store import VectorStore


# -------------------------------------------------
# Chunking tests
# -------------------------------------------------


def test_chunks_text():
    text = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence."
    )

    chunks = chunk_text(text=text)

    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) >= 20 for c in chunks)


def test_small_input_not_dropped():
    text = "Short meaningful sentence."
    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


# -------------------------------------------------
# process_text tests
# -------------------------------------------------


def test_process_text_no_chunks():
    embedding_model = Mock()
    embedding_model.model = True  # simulate loaded model

    vector_store = Mock(spec=VectorStore)
    vector_store.ntotal = 0
    vector_store.index = True

    result = process_text(
        text="   ",
        doc_id="doc1",
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    assert result["chunks_added"] == 0
    assert result["total_docs"] == 0
    embedding_model.embed_batch.assert_not_called()


def test_process_text_embedding_count_mismatch_is_graceful():
    embedding_model = Mock()
    embedding_model.embed_batch.return_value = [[0.1, 0.2]]  # fewer embeddings

    vector_store = Mock(spec=VectorStore)
    vector_store.index = True
    vector_store.ntotal = 0
    vector_store.add_embeddings = Mock()

    result = process_text(
        text="This is a valid chunkable sentence. " * 3,
        doc_id="doc_partial",
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    assert result["chunks_added"] == 1
    vector_store.add_embeddings.assert_called_once()
