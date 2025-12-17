from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from typing import Optional
from datetime import datetime


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    doc_id: Optional[str] = Field(None, max_length=100)
    chunk_size: int = Field(512, ge=100, le=2048)
    chunk_overlap: int = Field(50, ge=0)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info: ValidationInfo) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


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


# -------------------------------------------------
# Reusable OpenAPI response envelopes
# -------------------------------------------------


class IngestResponse(BaseModel):
    success: bool = True
    data: TextResponse


class StatusResponse(BaseModel):
    success: bool = True
    data: IndexStats
