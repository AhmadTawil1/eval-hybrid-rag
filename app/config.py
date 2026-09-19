from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    llm_model: str = "gpt-4o-mini"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
