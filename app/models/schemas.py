from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    doc_id: Optional[str] = Field(None, max_length=100)


class TextResponse(BaseModel):
    doc_id: str
    chunks_added: int
    total_docs: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IndexStats(BaseModel):
    total_documents: int
    embedding_dimension: int
    index_type: str


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AdminResetRequest(BaseModel):
    admin_key: str


# -------------------------------------------------
# Reusable OpenAPI response envelopes
# -------------------------------------------------


class IngestResponse(BaseModel):
    success: bool = True
    data: TextResponse


class StatusResponse(BaseModel):
    success: bool = True
    data: IndexStats
