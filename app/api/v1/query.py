from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List
from app.core.deps import get_embedding_model_dep, get_vector_store_dep
from app.core.config import settings
from app.services.retrieval_service import run_query
from app.services.generation_service import generate_answer
from app.services.embedding_service import EmbeddingModel
from app.services.vector_store import VectorStore
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(4, ge=1, le=20)


class ContextChunk(BaseModel):
    text: str
    score: float
    doc_id: str
    chunk_index: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    contexts: List[ContextChunk]
    total_chunks: int


@router.post("", response_model=QueryResponse)
async def run_rag_query(
    request: QueryRequest,
    embedding_model: EmbeddingModel = Depends(get_embedding_model_dep),
    vector_store: VectorStore = Depends(get_vector_store_dep),
):
    """
    Run RAG query: retrieve relevant chunks and generate answer using GPT-4.0-mini.
    """
    try:
        logger.info(f"Processing query: '{request.query[:50]}...'")

        # Step 1: Retrieve relevant context chunks
        contexts = run_query(
            query=request.query,
            top_k=request.top_k,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        logger.info(f"Retrieved {len(contexts)} context chunks")

        # Step 2: Generate answer using retrieved context
        answer = generate_answer(
            query=request.query, contexts=contexts, settings=settings
        )

        # Step 3: Format response
        formatted_contexts = [
            ContextChunk(
                text=chunk["text"],
                score=chunk["score"],
                doc_id=chunk["doc_id"],
                chunk_index=chunk["chunk_index"],
            )
            for chunk in contexts
        ]

        return QueryResponse(
            query=request.query,
            answer=answer,
            contexts=formatted_contexts,
            total_chunks=len(contexts),
        )

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@router.get("/health")
async def query_health():
    """Health check for query service."""
    return {"status": "healthy", "service": "query"}
