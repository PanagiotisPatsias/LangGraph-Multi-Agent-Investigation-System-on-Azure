from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Any


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # azure openai
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"

    # azure ai search
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index_name: str = "investigations-index"

    # azure cosmos db
    azure_cosmos_endpoint: str
    azure_cosmos_key: str
    azure_cosmos_database: str = "investigations"
    azure_cosmos_container: str = "investigation-states"

    # azure blob storage
    azure_storage_connection_string: str
    azure_storage_container: str = "investigation-documents"

    # web search
    bing_search_api_key: str = ""
    bing_search_endpoint: str = "https://api.bing.microsoft.com/v7.0/search"

    # application
    app_env: str = "development"
    log_level: str = "INFO"
    max_agent_iterations: int = 15

    # langfuse observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


@lru_cache
def get_langfuse_callback() -> Any | None:
    """Return a Langfuse CallbackHandler if credentials are configured, else None."""
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    return CallbackHandler()
