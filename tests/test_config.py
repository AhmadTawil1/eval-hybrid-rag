import pytest

from app import generation
from app.config import Settings


def test_settings_load_without_an_openai_key():
    settings = Settings(_env_file=None, openai_api_key=None)
    assert settings.openai_api_key is None


def test_qdrant_url_defaults_to_localhost_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert Settings(_env_file=None).qdrant_url == "http://localhost:6333"
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    assert Settings(_env_file=None).qdrant_url == "http://qdrant:6333"


def test_generation_needs_an_openai_key(monkeypatch):
    monkeypatch.setattr(generation, "get_settings", lambda: Settings(_env_file=None, openai_api_key=None))
    generation.get_openai_client.cache_clear()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        generation.get_openai_client()
