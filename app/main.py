import json
import logging
import pathlib
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app import pipeline
from app.config import get_settings
from app.embeddings import get_embedding_model
from app.generation import GenerationError, MissingApiKeyError
from app.limits import DailyBudget, SlidingWindowLimiter
from app.retrieval.dense import COLLECTION_NAME, VECTOR_SIZE, get_qdrant_client
from app.retrieval.sparse import get_bm25
from app.schemas import (
    MAX_QUERY_CHARS,
    LimitsResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)

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

STATIC_DIR = pathlib.Path(__file__).parent / "static"
REPORT_PATH = pathlib.Path("paper/experiment_report.pdf")

PAGE_CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Content-Security-Policy"] = PAGE_CSP
        response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/report.pdf", include_in_schema=False)
def report() -> FileResponse:
    if not REPORT_PATH.is_file():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(REPORT_PATH, media_type="application/pdf", content_disposition_type="inline")


for url_path, directory in (("/static", STATIC_DIR), ("/charts", pathlib.Path("eval/charts")), ("/assets", pathlib.Path("assets"))):
    if directory.is_dir():
        app.mount(url_path, StaticFiles(directory=directory), name=url_path.strip("/"))


@dataclass
class Limits:
    answers: SlidingWindowLimiter
    searches: SlidingWindowLimiter
    budget: DailyBudget


@lru_cache(maxsize=1)
def get_limits() -> Limits:
    s = get_settings()
    return Limits(
        answers=SlidingWindowLimiter(s.answers_per_visitor_per_hour, 3600),
        searches=SlidingWindowLimiter(s.searches_per_visitor_per_minute, 60),
        budget=DailyBudget(s.daily_answers_limit),
    )


def client_ip(request: Request) -> str:
    if get_settings().trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def too_many_requests(message: str, retry_after: int) -> HTTPException:
    return HTTPException(status_code=429, detail=message, headers={"Retry-After": str(retry_after)})


PAUSED_MESSAGE = (
    "Answer generation is paused for today because this demo's daily budget has been reached. "
    "Here are the most relevant passages instead."
)


def paused_response(body: QueryRequest, start: float) -> QueryResponse:
    passages = pipeline.search(body.query, body.mode)
    return QueryResponse(
        answer=PAUSED_MESSAGE,
        latency_ms=(time.perf_counter() - start) * 1000,
        citations=[],
        generation="paused_daily_limit",
        passages=passages,
    )


@app.post("/api/v1/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    limits = get_limits()
    start = time.perf_counter()

    if limits.budget.remaining() <= 0:
        return paused_response(body, start)

    allowed, retry_after = limits.answers.check(client_ip(request))
    if not allowed:
        minutes = -(-retry_after // 60)
        raise too_many_requests(
            f"Answer limit reached ({limits.answers.limit} per hour per visitor). Try again in about {minutes} min, "
            "or use the free search-only option (the \"Search only\" button on the page, or /api/v1/search), "
            "which shows the retrieved passages without an AI-written answer.",
            retry_after,
        )

    if not limits.budget.try_spend():
        return paused_response(body, start)

    try:
        result = pipeline.query(body.query, body.mode)
    except MissingApiKeyError:
        raise HTTPException(status_code=503, detail="Answer generation is not configured: OPENAI_API_KEY is not set.")
    except (GenerationError, ValidationError):
        logger.exception("generation failed")
        raise HTTPException(status_code=502, detail="The model returned an invalid answer. Please try again.")
    latency_ms = (time.perf_counter() - start) * 1000
    return QueryResponse(answer=result["answer"], latency_ms=latency_ms, citations=result["citations"])


@app.post("/api/v1/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request) -> SearchResponse:
    allowed, retry_after = get_limits().searches.check(client_ip(request))
    if not allowed:
        raise too_many_requests("Too many searches. Please slow down and try again shortly.", retry_after)
    start = time.perf_counter()
    results = pipeline.search(body.query, body.mode, body.k)
    return SearchResponse(mode=body.mode, latency_ms=(time.perf_counter() - start) * 1000, results=results)


@app.get("/api/v1/limits", response_model=LimitsResponse)
def limits_info() -> LimitsResponse:
    s = get_settings()
    return LimitsResponse(
        generation_enabled=bool(s.openai_api_key),
        answers_per_visitor_per_hour=s.answers_per_visitor_per_hour,
        daily_answers_limit=s.daily_answers_limit,
        daily_answers_remaining=get_limits().budget.remaining(),
        max_query_chars=MAX_QUERY_CHARS,
    )


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
