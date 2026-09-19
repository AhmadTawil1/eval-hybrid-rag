import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.generation import GenerationError, MissingApiKeyError
from app.schemas import MAX_QUERY_CHARS

# No `with` block: the lifespan (model/index loading) is not run, so no Qdrant or model is needed.
client = TestClient(main.app)


@pytest.fixture(autouse=True)
def fresh_limits():
    main.get_limits.cache_clear()
    yield
    main.get_limits.cache_clear()


def use_settings(monkeypatch, **overrides):
    settings = Settings(_env_file=None, openai_api_key="test", **overrides)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    main.get_limits.cache_clear()
    return settings


FAKE_PASSAGES = [
    {
        "rank": 1,
        "chunk_id": "abc123",
        "source_file": "Some article",
        "section_header": "Some article — Polygon",
        "url": "https://example.com/a",
        "text": "Passage text.",
        "score": 0.5,
    }
]

FAKE_RESULT = {
    "answer": "Jeppe Carlsen [abc123]",
    "citations": [{"chunk_id": "abc123", "source_file": "Some article", "section_header": "Some article — Polygon"}],
    "contexts": ["ctx"],
    "latency_ms": 1.0,
}


def test_invalid_mode_returns_422():
    response = client.post("/api/v1/query", json={"query": "hello", "mode": "fuzzy"})
    assert response.status_code == 422


def test_empty_query_returns_422():
    response = client.post("/api/v1/query", json={"query": "", "mode": "hybrid"})
    assert response.status_code == 422


def test_query_returns_answer_citations_and_latency(monkeypatch):
    monkeypatch.setattr(main.pipeline, "query", lambda query, mode: FAKE_RESULT)
    response = client.post("/api/v1/query", json={"query": "Who?", "mode": "dense"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == FAKE_RESULT["answer"]
    assert body["citations"] == FAKE_RESULT["citations"]
    assert isinstance(body["latency_ms"], float)


def test_query_defaults_to_hybrid_and_passes_the_mode(monkeypatch):
    seen = {}

    def fake_query(query, mode):
        seen.update(query=query, mode=mode)
        return FAKE_RESULT

    monkeypatch.setattr(main.pipeline, "query", fake_query)
    client.post("/api/v1/query", json={"query": "Who?"})
    assert seen == {"query": "Who?", "mode": "hybrid"}


def test_query_returns_502_when_the_model_output_is_invalid(monkeypatch):
    def broken(query, mode):
        raise GenerationError("no valid GroundedAnswer returned")

    monkeypatch.setattr(main.pipeline, "query", broken)
    response = client.post("/api/v1/query", json={"query": "Who?"})
    assert response.status_code == 502
    assert "GroundedAnswer" not in response.text


def test_query_returns_503_with_a_clear_message_when_the_key_is_missing(monkeypatch):
    def no_key(query, mode):
        raise MissingApiKeyError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(main.pipeline, "query", no_key)
    response = client.post("/api/v1/query", json={"query": "Who?"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_metrics_returns_the_results_json():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    body = response.json()
    assert set(body["modes"]) == {"dense", "sparse", "hybrid"}


def test_metrics_returns_404_when_the_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "RESULTS_PATH", tmp_path / "missing.json")
    assert client.get("/api/v1/metrics").status_code == 404


def test_healthz_returns_200_when_services_are_up(monkeypatch):
    monkeypatch.setattr(main, "qdrant_ok", lambda: True)
    monkeypatch.setattr(main, "embeddings_ok", lambda: True)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant": True, "embeddings": True}


@pytest.mark.parametrize("qdrant,embeddings", [(False, True), (True, False), (False, False)])
def test_healthz_returns_503_if_either_service_is_down(monkeypatch, qdrant, embeddings):
    monkeypatch.setattr(main, "qdrant_ok", lambda: qdrant)
    monkeypatch.setattr(main, "embeddings_ok", lambda: embeddings)
    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_query_longer_than_the_limit_returns_422():
    too_long = "x" * (MAX_QUERY_CHARS + 1)
    assert client.post("/api/v1/query", json={"query": too_long}).status_code == 422
    assert client.post("/api/v1/search", json={"query": too_long}).status_code == 422


def test_answer_limit_per_visitor_returns_429_with_retry_after(monkeypatch):
    use_settings(monkeypatch, answers_per_visitor_per_hour=2, daily_answers_limit=100)
    monkeypatch.setattr(main.pipeline, "query", lambda query, mode: FAKE_RESULT)
    assert client.post("/api/v1/query", json={"query": "Who?"}).status_code == 200
    assert client.post("/api/v1/query", json={"query": "Who?"}).status_code == 200
    blocked = client.post("/api/v1/query", json={"query": "Who?"})
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0
    assert "2 per hour" in blocked.json()["detail"]


def test_a_blocked_visitor_does_not_use_up_the_daily_budget(monkeypatch):
    settings = use_settings(monkeypatch, answers_per_visitor_per_hour=1, daily_answers_limit=5)
    monkeypatch.setattr(main.pipeline, "query", lambda query, mode: FAKE_RESULT)
    client.post("/api/v1/query", json={"query": "Who?"})
    client.post("/api/v1/query", json={"query": "Who?"})  # blocked
    assert main.get_limits().budget.remaining() == settings.daily_answers_limit - 1


def test_visitors_are_told_apart_by_the_forwarded_header_only_when_trusted(monkeypatch):
    monkeypatch.setattr(main.pipeline, "query", lambda query, mode: FAKE_RESULT)

    use_settings(monkeypatch, answers_per_visitor_per_hour=1, trust_proxy_headers=True)
    first = client.post("/api/v1/query", json={"query": "Who?"}, headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.1"})
    other = client.post("/api/v1/query", json={"query": "Who?"}, headers={"X-Forwarded-For": "2.2.2.2"})
    again = client.post("/api/v1/query", json={"query": "Who?"}, headers={"X-Forwarded-For": "1.1.1.1"})
    assert (first.status_code, other.status_code, again.status_code) == (200, 200, 429)

    use_settings(monkeypatch, answers_per_visitor_per_hour=1, trust_proxy_headers=False)
    spoofed = [
        client.post("/api/v1/query", json={"query": "Who?"}, headers={"X-Forwarded-For": f"9.9.9.{i}"}).status_code
        for i in range(2)
    ]
    assert spoofed == [200, 429]  # the header is ignored, so changing it does not dodge the limit


def test_when_the_daily_budget_is_spent_query_falls_back_to_passages_without_calling_the_llm(monkeypatch):
    use_settings(monkeypatch, daily_answers_limit=0)

    def must_not_run(query, mode):
        raise AssertionError("the LLM pipeline must not run when the daily budget is spent")

    monkeypatch.setattr(main.pipeline, "query", must_not_run)
    monkeypatch.setattr(main.pipeline, "search", lambda query, mode, k=5: FAKE_PASSAGES)
    response = client.post("/api/v1/query", json={"query": "Who?", "mode": "sparse"})
    assert response.status_code == 200
    body = response.json()
    assert body["generation"] == "paused_daily_limit"
    assert body["citations"] == []
    assert body["passages"][0]["chunk_id"] == "abc123"
    assert "daily budget" in body["answer"]


def test_the_fallback_is_free_for_visitors_over_their_hourly_limit(monkeypatch):
    use_settings(monkeypatch, daily_answers_limit=0, answers_per_visitor_per_hour=1)
    monkeypatch.setattr(main.pipeline, "search", lambda query, mode, k=5: FAKE_PASSAGES)
    codes = [client.post("/api/v1/query", json={"query": "Who?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 200]


def test_a_normal_answer_reports_generation_ok(monkeypatch):
    use_settings(monkeypatch)
    monkeypatch.setattr(main.pipeline, "query", lambda query, mode: FAKE_RESULT)
    body = client.post("/api/v1/query", json={"query": "Who?"}).json()
    assert body["generation"] == "ok"
    assert body["passages"] == []


def test_search_returns_ranked_passages_without_an_llm(monkeypatch):
    seen = {}

    def fake_search(query, mode, k=5):
        seen.update(query=query, mode=mode, k=k)
        return FAKE_PASSAGES

    monkeypatch.setattr(main.pipeline, "search", fake_search)
    monkeypatch.setattr(main.pipeline, "query", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no LLM")))
    response = client.post("/api/v1/search", json={"query": "Who?", "mode": "dense", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "dense"
    assert body["results"][0]["rank"] == 1 and body["results"][0]["url"] == "https://example.com/a"
    assert seen == {"query": "Who?", "mode": "dense", "k": 3}


@pytest.mark.parametrize("payload", [{"query": "q", "k": 0}, {"query": "q", "k": 11}, {"query": "q", "mode": "fuzzy"}])
def test_search_rejects_invalid_input(payload):
    assert client.post("/api/v1/search", json=payload).status_code == 422


def test_search_has_its_own_generous_rate_limit(monkeypatch):
    use_settings(monkeypatch, searches_per_visitor_per_minute=2)
    monkeypatch.setattr(main.pipeline, "search", lambda query, mode, k=5: FAKE_PASSAGES)
    codes = [client.post("/api/v1/search", json={"query": "Who?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_limits_endpoint_reports_the_configuration_and_what_is_left(monkeypatch):
    use_settings(monkeypatch, answers_per_visitor_per_hour=5, daily_answers_limit=80)
    monkeypatch.setattr(main.pipeline, "query", lambda query, mode: FAKE_RESULT)
    client.post("/api/v1/query", json={"query": "Who?"})
    body = client.get("/api/v1/limits").json()
    assert body == {
        "generation_enabled": True,
        "answers_per_visitor_per_hour": 5,
        "daily_answers_limit": 80,
        "daily_answers_remaining": 79,
        "max_query_chars": MAX_QUERY_CHARS,
    }
