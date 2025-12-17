from unittest.mock import Mock

from app.services.ingestion_service import (
    process_text,
    sentence_aware_chunk_text,
    fixed_chunk_text,
)
from app.services.vector_store import VectorStore


# -------------------------------------------------
# Chunking tests
# -------------------------------------------------


def test_sentence_aware_chunking_basic():
    text = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence."
    )

    chunks = sentence_aware_chunk_text(
        text=text,
        chunk_size=60,
        chunk_overlap=0,
    )

    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) >= 20 for c in chunks)


def test_fixed_chunking_basic():
    text = "A" * 200

    chunks = fixed_chunk_text(
        text=text,
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_chunking_empty_text():
    chunks = sentence_aware_chunk_text(
        text="   ",
        chunk_size=100,
        chunk_overlap=10,
    )
    assert chunks == []


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
        chunk_size=100,
        chunk_overlap=10,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    assert result["chunks_added"] == 0
    assert result["total_docs"] == 0
    embedding_model.embed_batch.assert_not_called()


def test_process_text_happy_path_sentence_chunking():
    text = "This is sentence one. " "This is sentence two. " "This is sentence three."

    embedding_model = Mock()
    embedding_model.model = True
    embedding_model.embed_batch.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    vector_store = Mock(spec=VectorStore)
    vector_store.index = True
    vector_store.ntotal = 0

    def add_embeddings_side_effect(embeddings, metadatas):
        vector_store.ntotal += len(embeddings)

    vector_store.add_embeddings.side_effect = add_embeddings_side_effect

    result = process_text(
        text=text,
        doc_id="doc123",
        chunk_size=50,
        chunk_overlap=0,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunking_strategy="sentence",
    )

    assert result["chunks_added"] == 2
    assert result["total_docs"] == 2

    embedding_model.embed_batch.assert_called_once()
    vector_store.add_embeddings.assert_called_once()

    embeddings_arg, metadata_arg = vector_store.add_embeddings.call_args[0]

    assert len(embeddings_arg) == len(metadata_arg)
    assert metadata_arg[0]["doc_id"] == "doc123"
    assert metadata_arg[0]["chunk_index"] == 0
    assert metadata_arg[0]["chunking_strategy"] == "sentence"


def test_process_text_fixed_chunking():
    text = "A" * 300

    embedding_model = Mock()
    embedding_model.model = True
    embedding_model.embed_batch.return_value = [[0.1, 0.2]] * 5

    vector_store = Mock(spec=VectorStore)
    vector_store.index = True
    vector_store.ntotal = 0

    def add_embeddings_side_effect(embeddings, metadatas):
        vector_store.ntotal += len(embeddings)

    vector_store.add_embeddings.side_effect = add_embeddings_side_effect

    result = process_text(
        text=text,
        doc_id="doc_fixed",
        chunk_size=80,
        chunk_overlap=20,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunking_strategy="fixed",
    )

    assert result["chunks_added"] > 0
    assert result["total_docs"] == result["chunks_added"]

    embedding_model.embed_batch.assert_called_once()
    vector_store.add_embeddings.assert_called_once()


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
        chunk_size=50,
        chunk_overlap=0,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    assert result["chunks_added"] == 1
    vector_store.add_embeddings.assert_called_once()
