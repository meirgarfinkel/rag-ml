from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1 import ingest, query, health
from app.core.config import settings
from app.core.deps import set_embedding_model, set_vector_store
from app.services.embedding_service import load_embedding_model
from app.services.vector_store import load_vector_store, save_vector_store

# -------------------------------------------------
# Logging
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Lifespan (startup / shutdown)
# -------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    """
    vector_store = None

    logger.info("Starting RAG-ML application")

    try:
        logger.info("Loading embedding model")
        embedding_model = load_embedding_model(settings)

        logger.info("Loading vector store")
        vector_store, _ = load_vector_store(settings)

        # Register global dependencies
        set_embedding_model(embedding_model)
        set_vector_store(vector_store)

        logger.info("Application startup complete")
        yield

    except Exception:
        logger.exception("Fatal error during application startup")
        raise

    finally:
        logger.info("Shutting down application")

        if vector_store is not None:
            try:
                logger.info("Saving vector store")
                save_vector_store(vector_store, vector_store.metadata, settings)
            except Exception:
                logger.exception("Failed to save vector store during shutdown")

        logger.info("Shutdown complete")


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------

app = FastAPI(
    title="RAG-ML",
    version="0.0.1",
    description="Retrieval-Augmented Generation demo (FAISS + GPT-4o-mini)",
    lifespan=lifespan,
)

# -------------------------------------------------
# API routes
# -------------------------------------------------

app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])

# -------------------------------------------------
# Frontend (static)
# -------------------------------------------------

app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend",
)

# -------------------------------------------------
# Local dev entrypoint
# -------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
