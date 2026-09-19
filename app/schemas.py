from typing import Literal

from pydantic import BaseModel, Field

MAX_QUERY_CHARS = 500

Mode = Literal["dense", "sparse", "hybrid"]


class GroundedAnswer(BaseModel):
    answer: str
    referenced_chunks: list[str]


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    section_header: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    mode: Mode = "hybrid"


class Passage(BaseModel):
    rank: int
    chunk_id: str
    source_file: str
    section_header: str
    url: str
    text: str
    score: float = Field(description="Mode-specific (cosine, BM25 or RRF): compare ranks, not scores across modes.")


class QueryResponse(BaseModel):
    answer: str
    latency_ms: float
    citations: list[Citation]
    generation: Literal["ok", "paused_daily_limit"] = "ok"
    passages: list[Passage] = Field(
        default_factory=list, description="Filled only when generation is paused: the retrieved passages, no LLM answer."
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    mode: Mode = "hybrid"
    k: int = Field(default=5, ge=1, le=10)


class SearchResponse(BaseModel):
    mode: Mode
    latency_ms: float
    results: list[Passage]


class LimitsResponse(BaseModel):
    generation_enabled: bool
    answers_per_visitor_per_hour: int
    daily_answers_limit: int
    daily_answers_remaining: int
    max_query_chars: int
