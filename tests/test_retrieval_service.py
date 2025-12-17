import pytest
from unittest.mock import Mock

from app.services.retrieval_service import run_query, normalize_l2_scores
from app.services.vector_store import VectorStore


# -------------------------------------------------
# normalize_l2_scores tests
# -------------------------------------------------


def test_normalize_l2_scores_basic():
    results = [
        {"score": 0.2},
        {"score": 0.4},
        {"score": 0.6},
    ]

    normalize_l2_scores(results)

    scores = [r["score"] for r in results]
    assert max(scores) == 1.0
    assert min(scores) == 0.0


def test_normalize_l2_scores_equal_distances():
    results = [
        {"score": 0.5},
        {"score": 0.5},
    ]

    normalize_l2_scores(results)

    assert all(r["score"] == 1.0 for r in results)


def test_normalize_l2_scores_empty():
    normalize_l2_scores([])  # should not raise


# -------------------------------------------------
# run_query tests
# -------------------------------------------------


def test_run_query_empty_query():
    embedding_model = Mock()
    vector_store = Mock()

    results = run_query(
        query="   ",
        top_k=3,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    assert results == []


def test_run_query_model_not_loaded():
    embedding_model = Mock()
    embedding_model.model = None

    vector_store = Mock(spec=VectorStore)
    vector_store.index = True

    with pytest.raises(RuntimeError):
        run_query(
            query="hello",
            top_k=3,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )


def test_run_query_no_index():
    embedding_model = Mock()
    embedding_model.model = True
    embedding_model.embed.return_value = [0.1, 0.2]

    vector_store = Mock(spec=VectorStore)
    vector_store.index = None

    results = run_query(
        query="hello",
        top_k=3,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    assert results == []


def test_run_query_happy_path():
    embedding_model = Mock()
    embedding_model.model = True
    embedding_model.embed.return_value = [0.1, 0.2]

    vector_store = Mock(spec=VectorStore)
    vector_store.index = True
    vector_store.search.return_value = [
        {"text": "chunk1", "score": 0.2},
        {"text": "chunk2", "score": 0.6},
    ]

    results = run_query(
        query="test query",
        top_k=2,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    assert len(results) == 2
    assert results[0]["score"] > results[1]["score"]
    embedding_model.embed.assert_called_once()
    vector_store.search.assert_called_once()


def test_run_query_search_returns_empty():
    embedding_model = Mock()
    embedding_model.model = True
    embedding_model.embed.return_value = [0.1, 0.2]

    vector_store = Mock(spec=VectorStore)
    vector_store.index = True
    vector_store.search.return_value = []

    results = run_query(
        query="test",
        top_k=3,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    assert results == []
