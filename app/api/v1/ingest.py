import logging
import hashlib
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import get_embedding_model_dep, get_vector_store_dep
from app.services.ingestion_service import process_text
from app.services.embedding_service import EmbeddingModel
from app.services.vector_store import VectorStore
from app.models.schemas import TextResponse, IndexStats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["ingest"])


# -------------------------------------------------
# Schemas
# -------------------------------------------------


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    doc_id: Optional[str] = None


# -------------------------------------------------
# Helpers
# -------------------------------------------------


def _stable_doc_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"doc-{digest}"


# -------------------------------------------------
# Routes
# -------------------------------------------------


@router.post("/text", response_model=TextResponse)
async def ingest_text(
    request: TextRequest,
    embedding_model: EmbeddingModel = Depends(get_embedding_model_dep),
    vector_store: VectorStore = Depends(get_vector_store_dep),
):
    """
    Ingest raw text: chunk, embed, and store in FAISS.
    """
    try:
        doc_id = request.doc_id or _stable_doc_id(request.text)

        logger.info(f"Ingesting document '{doc_id}'")

        result = process_text(
            text=request.text,
            doc_id=doc_id,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        logger.info(
            f"Ingestion complete for '{doc_id}': "
            f"{result['chunks_added']} chunks added"
        )

        return TextResponse(
            doc_id=doc_id,
            chunks_added=result["chunks_added"],
            total_docs=result["total_docs"],
        )

    except Exception as e:
        logger.exception("Text ingestion failed")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.post("/demo", response_model=TextResponse)
async def ingest_demo(
    embedding_model: EmbeddingModel = Depends(get_embedding_model_dep),
    vector_store: VectorStore = Depends(get_vector_store_dep),
):
    """
    Load and ingest a predefined demo dataset.
    """
    demo_path = Path(__file__).resolve().parents[3] / "data" / "demo_dataset.txt"

    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="Demo dataset not found")

    try:
        logger.info("Ingesting demo dataset")

        demo_text = demo_path.read_text(encoding="utf-8")

        result = process_text(
            text=demo_text,
            doc_id="demo-dataset",
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        return TextResponse(
            doc_id="demo-dataset",
            chunks_added=result["chunks_added"],
            total_docs=result["total_docs"],
        )

    except Exception as e:
        logger.exception("Demo ingestion failed")
        raise HTTPException(
            status_code=500,
            detail=f"Demo ingestion failed: {str(e)}",
        )


@router.get("/status", response_model=IndexStats)
async def get_index_status(
    vector_store: VectorStore = Depends(get_vector_store_dep),
):
    """
    Get FAISS index statistics.
    """
    try:
        return IndexStats(
            total_documents=vector_store.ntotal,
            embedding_dimension=vector_store.dimension,
            index_type=vector_store.index.__class__.__name__
            if vector_store.index
            else "None",
        )

    except Exception as e:
        logger.exception("Index status check failed")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}",
        )
