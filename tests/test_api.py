import pytest
from fastapi.testclient import TestClient

from app import main
from app.generation import GenerationError, MissingApiKeyError

# No `with` block: the lifespan (model/index loading) is not run, so no Qdrant or model is needed.
client = TestClient(main.app)

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
