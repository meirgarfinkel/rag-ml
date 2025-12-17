import numpy as np
import pytest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from app.services.embedding_service import (
    EmbeddingModel,
    load_embedding_model,
    unload_embedding_model,
)


@pytest.fixture
def settings():
    return SimpleNamespace(embedding_model_name="fake-model")


@patch("app.services.embedding_service.SentenceTransformer")
def test_model_loads_and_sets_dimension(mock_st):
    mock_instance = Mock()
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_st.return_value = mock_instance

    model = EmbeddingModel("test-model")
    model.load()

    assert model.model is mock_instance
    assert model.dimension == 384


def test_embed_fails_if_not_loaded():
    model = EmbeddingModel("test")

    with pytest.raises(RuntimeError):
        model.embed("hello")


@patch("app.services.embedding_service.SentenceTransformer")
def test_embed_single_text(mock_st):
    mock_instance = Mock()
    mock_instance.get_sentence_embedding_dimension.return_value = 3
    mock_instance.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_st.return_value = mock_instance

    model = EmbeddingModel("test-model")
    model.load()

    result = model.embed("hello world")

    assert result == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)


@patch("app.services.embedding_service.SentenceTransformer")
def test_embed_batch(mock_st):
    mock_instance = Mock()
    mock_instance.get_sentence_embedding_dimension.return_value = 2
    mock_instance.encode.return_value = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )
    mock_st.return_value = mock_instance

    model = EmbeddingModel("test-model")
    model.load()

    result = model.embed_batch(["a", "b"])

    assert len(result) == 2
    assert result[0] == pytest.approx([0.1, 0.2], rel=1e-6)
    assert result[1] == pytest.approx([0.3, 0.4], rel=1e-6)


def test_embed_rejects_empty_text():
    model = EmbeddingModel("test")
    model.model = Mock()

    with pytest.raises(ValueError):
        model.embed("")


def test_embed_batch_rejects_empty_list():
    model = EmbeddingModel("test")
    model.model = Mock()

    with pytest.raises(ValueError):
        model.embed_batch([])


@patch("app.services.embedding_service.EmbeddingModel")
def test_load_embedding_model(mock_model, settings):
    instance = Mock()
    mock_model.return_value = instance

    result = load_embedding_model(settings)

    mock_model.assert_called_once_with("fake-model")
    instance.load.assert_called_once()
    assert result == instance


def test_unload_embedding_model():
    model = EmbeddingModel("test")
    model.model = Mock()
    model.dimension = 384

    unload_embedding_model(model)

    assert model.model is None
    assert model.dimension is None
