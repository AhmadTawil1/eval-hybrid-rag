import json
import logging
import pathlib
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app import pipeline
from app.embeddings import get_embedding_model
from app.generation import GenerationError
from app.retrieval.dense import COLLECTION_NAME, VECTOR_SIZE, get_qdrant_client
from app.retrieval.sparse import get_bm25
from app.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

RESULTS_PATH = pathlib.Path("eval/results.json")


def load_resources() -> None:
    get_embedding_model()
    get_bm25()
    get_qdrant_client()
    pipeline.get_chunks_by_id()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(title="eval-hybrid-rag", lifespan=lifespan)


@app.post("/api/v1/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    try:
        result = pipeline.query(request.query, request.mode)
    except (GenerationError, ValidationError):
        logger.exception("generation failed")
        raise HTTPException(status_code=502, detail="The model returned an invalid answer. Please try again.")
    latency_ms = (time.perf_counter() - start) * 1000
    return QueryResponse(answer=result["answer"], latency_ms=latency_ms, citations=result["citations"])


@app.get("/api/v1/metrics")
def metrics() -> dict:
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No benchmark results found.")


def qdrant_ok() -> bool:
    try:
        return get_qdrant_client().collection_exists(COLLECTION_NAME)
    except Exception:
        logger.warning("healthz: qdrant check failed", exc_info=True)
        return False


def embeddings_ok() -> bool:
    try:
        vector = get_embedding_model().encode("health check", normalize_embeddings=True)
        return len(vector) == VECTOR_SIZE
    except Exception:
        logger.warning("healthz: embedding model check failed", exc_info=True)
        return False


@app.get("/healthz")
def healthz() -> JSONResponse:
    checks = {"qdrant": qdrant_ok(), "embeddings": embeddings_ok()}
    healthy = all(checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "unavailable", **checks},
    )
