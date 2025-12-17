from fastapi import APIRouter, Depends, HTTPException
from app.core.deps import get_embedding_model_dep, get_vector_store_dep
from app.services.embedding_service import EmbeddingModel
from app.services.vector_store import VectorStore
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["health"])


@router.get("/")
async def health_check(
    embedding_model: EmbeddingModel = Depends(get_embedding_model_dep),
    vector_store: VectorStore = Depends(get_vector_store_dep),
):
    """
    Comprehensive health check for RAG system.
    """
    try:
        status = {
            "status": "healthy",
            "service": "health",
            "timestamp": "2025-12-16T11:25:00Z",
            "embedding_model": {
                "loaded": embedding_model.model is not None,
                "model_name": embedding_model.model_name,
                "dimension": embedding_model.model.get_sentence_embedding_dimension()
                if embedding_model.model
                else 0,
            },
            "vector_store": {
                "loaded": vector_store.index is not None,
                "total_vectors": vector_store.ntotal,
                "dimension": vector_store.dimension,
                "index_type": type(vector_store.index).__name__
                if vector_store.index
                else "none",
            },
        }
        return status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/ready")
async def readiness_check():
    """Simple readiness probe."""
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Simple liveness probe."""
    return {"status": "alive"}
