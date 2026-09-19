from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    llm_model: str = "gpt-5-mini"
    max_output_tokens: int = 4000
    answers_per_visitor_per_hour: int = 5
    daily_answers_limit: int = 80
    searches_per_visitor_per_minute: int = 60
    trust_proxy_headers: bool = False
    judge_model: str = "gpt-5.6-terra"
    judge_embedding_model: str = "text-embedding-3-small"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
