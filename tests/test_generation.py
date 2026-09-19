import json

import httpx
import pytest
from openai import OpenAI

from app import generation
from app.config import Settings
from app.generation import GenerationError, generate

CHUNKS = [{"chunk_id": "abc123", "text": "The trial began in October."}]


def fake_openai(seen: dict, finish_reason: str = "stop", content: str | None = None) -> OpenAI:
    content = content or json.dumps({"answer": "October [abc123]", "referenced_chunks": ["abc123"]})

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        message = {"role": "assistant", "content": content, "refusal": None}
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
            },
        )

    return OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def use_settings(monkeypatch, **overrides):
    settings = Settings(_env_file=None, openai_api_key="test", **overrides)
    monkeypatch.setattr(generation, "get_settings", lambda: settings)


def test_request_carries_the_output_token_cap(monkeypatch):
    use_settings(monkeypatch, llm_model="gpt-5-mini", max_output_tokens=1234)
    seen = {}
    generate("When?", CHUNKS, client=fake_openai(seen))
    assert seen["max_completion_tokens"] == 1234
    assert seen["response_format"]["type"] == "json_schema"


def test_gpt5_models_get_no_temperature(monkeypatch):
    use_settings(monkeypatch, llm_model="gpt-5-mini")
    seen = {}
    generate("When?", CHUNKS, client=fake_openai(seen))
    assert "temperature" not in seen


def test_other_models_get_temperature_zero(monkeypatch):
    use_settings(monkeypatch, llm_model="gpt-4o-mini")
    seen = {}
    generate("When?", CHUNKS, client=fake_openai(seen))
    assert seen["temperature"] == 0


def test_an_answer_cut_off_by_the_token_cap_raises_generation_error(monkeypatch):
    use_settings(monkeypatch, llm_model="gpt-5-mini", max_output_tokens=10)
    truncated = fake_openai({}, finish_reason="length", content='{"answer": "Oct')
    with pytest.raises(GenerationError, match="token limit"):
        generate("When?", CHUNKS, client=truncated)
