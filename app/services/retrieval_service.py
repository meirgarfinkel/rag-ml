import logging
from typing import List, Dict, Any

from app.services.embedding_service import EmbeddingModel
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def normalize_l2_scores(results: List[Dict[str, Any]]) -> None:
    """
    Convert L2 distances to similarity scores in-place.
    Higher score = more similar (0.0–1.0).
    """
    if not results:
        return

    distances = [
        r.get("score") for r in results if isinstance(r.get("score"), (int, float))
    ]

    if not distances:
        return

    max_distance = max(distances)
    min_distance = min(distances)

    # Avoid division by zero (all distances equal)
    if max_distance == min_distance:
        for r in results:
            r["score"] = 1.0
        return

    for r in results:
        d = r["score"]
        similarity = 1.0 - (d - min_distance) / (max_distance - min_distance)
        r["score"] = float(similarity)


def run_query(
    query: str,
    top_k: int,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most relevant chunks for a query.
    """
    if not query or not query.strip():
        logger.warning("Empty query received")
        return []

    if embedding_model.model is None:
        raise RuntimeError("Embedding model not loaded")

    if vector_store.index is None:
        logger.warning("Vector store not initialized")
        return []

    logger.info(f"Running query search for top-{top_k}: '{query[:50]}...'")

    # Step 1: Embed query
    query_embedding = embedding_model.embed(query)

    if not query_embedding:
        logger.warning("Query embedding is empty")
        return []

    # Step 2: Vector search
    results = vector_store.search(query_embedding, top_k)

    # Step 3: Normalize scores
    normalize_l2_scores(results)

    logger.info(f"Found {len(results)} relevant chunks")
    return results
