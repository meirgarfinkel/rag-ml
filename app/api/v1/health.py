from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["health"])

@router.get("")
def health():
    return {"status": "ok"}

@router.get("/live")
def live():
    return {"status": "alive"}

@router.get("/ready")
def ready():
    return {"status": "ready"}
