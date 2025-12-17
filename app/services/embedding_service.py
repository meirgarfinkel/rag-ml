import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Safe wrapper for sentence-transformers embedding models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.dimension: Optional[int] = None

    def load(self) -> None:
        """Load the embedding model into memory."""
        if self.model is not None:
            return

        logger.info("Loading embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded (dim=%d)", self.dimension)

    def _ensure_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load() first.")

    def embed(self, text: str) -> List[float]:
        """Embed a single text into a vector."""
        self._ensure_loaded()

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        if embedding.ndim != 1:
            raise RuntimeError("Expected 1D embedding for single text")

        return embedding.astype(np.float32).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        self._ensure_loaded()

        if not texts:
            raise ValueError("texts list cannot be empty")

        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("All texts must be non-empty strings")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        if embeddings.ndim != 2:
            raise RuntimeError("Expected 2D embedding matrix for batch input")

        return embeddings.astype(np.float32).tolist()


def load_embedding_model(settings: Settings) -> EmbeddingModel:
    """Factory for application startup."""
    model = EmbeddingModel(settings.embedding_model_name)
    model.load()
    return model


def unload_embedding_model(model: Optional[EmbeddingModel]) -> None:
    """
    Explicit unload hook.
    SentenceTransformers does not support true unloading,
    but this allows GC and future extensibility.
    """
    if model is not None:
        logger.info("Unloading embedding model: %s", model.model_name)
        model.model = None
        model.dimension = None
