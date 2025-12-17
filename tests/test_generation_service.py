from unittest.mock import Mock, patch

from app.services.generation_service import (
    generate_answer,
    build_rag_prompt,
)


# -------------------------------------------------
# build_rag_prompt tests
# -------------------------------------------------


def test_build_rag_prompt_basic():
    query = "What is RAG?"
    contexts = [
        {"text": "RAG stands for Retrieval-Augmented Generation."},
        {"text": "It combines retrieval and generation."},
    ]

    prompt = build_rag_prompt(query, contexts)

    assert "What is RAG?" in prompt
    assert "Retrieval-Augmented Generation" in prompt
    assert "Context:" in prompt
    assert "Answer:" in prompt


def test_build_rag_prompt_ignores_missing_text():
    prompt = build_rag_prompt(
        "test",
        [{"foo": "bar"}, {"text": ""}],
    )

    assert "Context:" in prompt  # still valid
    assert "test" in prompt


# -------------------------------------------------
# generate_answer tests
# -------------------------------------------------


def test_generate_answer_empty_query():
    settings = Mock()
    result = generate_answer("", [], settings)

    assert "don't have enough information" in result.lower()


@patch("app.services.generation_service.openai.OpenAI")
def test_generate_answer_happy_path(mock_openai):
    settings = Mock()
    settings.openai_api_key = "fake-key"

    mock_client = Mock()
    mock_openai.return_value = mock_client

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="This is the answer."))]

    mock_client.chat.completions.create.return_value = mock_response

    result = generate_answer(
        query="What is Python?",
        contexts=[{"text": "Python is a programming language."}],
        settings=settings,
    )

    assert result == "This is the answer."
    mock_client.chat.completions.create.assert_called_once()


@patch("app.services.generation_service.openai.OpenAI")
def test_generate_answer_openai_error(mock_openai):
    settings = Mock()
    settings.openai_api_key = "fake-key"

    mock_client = Mock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")

    result = generate_answer(
        query="test",
        contexts=[{"text": "context"}],
        settings=settings,
    )

    assert "error" in result.lower()


@patch("app.services.generation_service.openai.OpenAI")
def test_generate_answer_no_context(mock_openai):
    settings = Mock()
    settings.openai_api_key = "fake-key"

    mock_client = Mock()
    mock_openai.return_value = mock_client

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="I don't know."))]
    mock_client.chat.completions.create.return_value = mock_response

    result = generate_answer(
        query="test",
        contexts=[],
        settings=settings,
    )

    assert result == "I don't know."
