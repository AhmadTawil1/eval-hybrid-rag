from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    llm_model: str = "gpt-5-mini"
    judge_model: str = "gpt-5.6-terra"
    judge_embedding_model: str = "text-embedding-3-small"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
