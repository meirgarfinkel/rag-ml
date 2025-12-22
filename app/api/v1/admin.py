from fastapi import APIRouter, Depends, HTTPException, status
from app.core.config import settings
from app.core.deps import get_vector_store_dep
from app.models.schemas import AdminResetRequest
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

@router.post("/reset", include_in_schema=False)
def reset_knowledge_base(
    payload: AdminResetRequest,
    vector_store: VectorStore = Depends(get_vector_store_dep),
):
    if payload.admin_key != settings.admin_reset_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key",
        )

    vector_store.reset()

    return {"success": True, "message": "Knowledge base reset"}
