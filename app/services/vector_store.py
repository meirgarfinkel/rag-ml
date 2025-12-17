import faiss
import pickle
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
from app.core.config import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS vector store using cosine similarity.

    Implementation details:
    - Embeddings are L2-normalized
    - FAISS IndexFlatIP is used (inner product == cosine similarity)
    - Higher score is better
    """

    REQUIRED_METADATA_KEYS = {"text", "doc_id", "chunk_index"}

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension: int = 0

    def create_index(self, dimension: int) -> None:
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatIP(dimension)
        self.dimension = dimension
        self.metadata = []
        logger.info("Created new FAISS IndexFlatIP(dimension=%d)", dimension)

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add embeddings and metadata to the index."""
        if self.index is None:
            raise RuntimeError("Index not initialized")

        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadata length mismatch")

        embedding_array = np.array(embeddings, dtype=np.float32)

        if embedding_array.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if embedding_array.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding_array.shape[1]} "
                f"does not match index dimension {self.dimension}"
            )

        # Validate metadata schema
        for meta in metadatas:
            if not self.REQUIRED_METADATA_KEYS.issubset(meta):
                raise ValueError(
                    f"Metadata missing required keys: {self.REQUIRED_METADATA_KEYS}"
                )

        # Normalize in-place (required for cosine similarity)
        faiss.normalize_L2(embedding_array)

        self.index.add(embedding_array)
        self.metadata.extend(metadatas)

        logger.info("Added %d embeddings (total=%d)", len(embeddings), self.ntotal)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Search for top-k most similar embeddings."""
        if self.index is None or self.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)

        if query_vec.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vec.shape[1]} "
                f"does not match index dimension {self.dimension}"
            )

        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            results.append(
                {
                    "text": meta["text"],
                    "score": float(score),  # Higher is better (cosine similarity)
                    "doc_id": meta["doc_id"],
                    "chunk_index": meta["chunk_index"],
                    "metadata": meta,
                }
            )

        return results

    @property
    def ntotal(self) -> int:
        """Total number of vectors in the index."""
        return self.index.ntotal if self.index else 0


def load_vector_store(settings: Settings) -> Tuple[VectorStore, List[Dict[str, Any]]]:
    """Load vector store from disk or initialize a new one."""
    vectorstore_path = Path(settings.vectorstore_path)
    index_path = vectorstore_path / "index.faiss"
    metadata_path = vectorstore_path / "metadata.pkl"

    vectorstore_path.mkdir(parents=True, exist_ok=True)
    store = VectorStore()

    if index_path.exists() and metadata_path.exists():
        try:
            store.index = faiss.read_index(str(index_path))
            store.dimension = store.index.d

            with open(metadata_path, "rb") as f:
                store.metadata = pickle.load(f)

            if store.ntotal != len(store.metadata):
                raise ValueError("Index / metadata count mismatch")

            logger.info("Loaded FAISS index with %d vectors", store.ntotal)
            return store, store.metadata

        except Exception as e:
            logger.warning("Corrupted vector store detected: %s", e)
            index_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)

    # Default initialization (empty store)
    DEFAULT_DIMENSION = 384  # all-MiniLM-L6-v2
    store.create_index(DEFAULT_DIMENSION)
    logger.info("Initialized empty vector store (dimension=%d)", DEFAULT_DIMENSION)

    return store, store.metadata


def save_vector_store(store: VectorStore, settings: Settings) -> None:
    """Persist vector store to disk."""
    if store.index is None or store.ntotal == 0:
        logger.warning("Empty index - skipping save")
        return

    vectorstore_path = Path(settings.vectorstore_path)
    vectorstore_path.mkdir(parents=True, exist_ok=True)

    index_path = vectorstore_path / "index.faiss"
    metadata_path = vectorstore_path / "metadata.pkl"

    faiss.write_index(store.index, str(index_path))

    with open(metadata_path, "wb") as f:
        pickle.dump(store.metadata, f)

    logger.info(
        "Saved vector store with %d vectors to %s",
        store.ntotal,
        vectorstore_path,
    )
