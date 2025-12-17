from typing import Optional
from app.services.embedding_service import EmbeddingModel
from app.services.vector_store import VectorStore

# Private global state (initialized at startup)
_embedding_model: Optional[EmbeddingModel] = None
_vector_store: Optional[VectorStore] = None


# ---------- Startup setters ----------


def set_embedding_model(model: EmbeddingModel) -> None:
    """Set embedding model during application startup."""
    global _embedding_model
    _embedding_model = model


def set_vector_store(store: VectorStore) -> None:
    """Set vector store during application startup."""
    global _vector_store
    _vector_store = store


# ---------- Dependency getters ----------


def get_embedding_model() -> EmbeddingModel:
    """Dependency getter for embedding model."""
    if _embedding_model is None:
        raise RuntimeError(
            "Embedding model not initialized. Startup event did not run."
        )
    return _embedding_model


def get_vector_store() -> VectorStore:
    """Dependency getter for vector store."""
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized. Startup event did not run.")
    return _vector_store


# ---------- FastAPI dependency wrappers ----------


def get_embedding_model_dep() -> EmbeddingModel:
    return get_embedding_model()


def get_vector_store_dep() -> VectorStore:
    return get_vector_store()
