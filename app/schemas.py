from typing import Literal

from pydantic import BaseModel, Field


class GroundedAnswer(BaseModel):
    answer: str
    referenced_chunks: list[str]


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    section_header: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"


class QueryResponse(BaseModel):
    answer: str
    latency_ms: float
    citations: list[Citation]
