from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"

    # Azure AI Search
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index_name: str = "investigations-index"

    # Azure Cosmos DB
    azure_cosmos_endpoint: str
    azure_cosmos_key: str
    azure_cosmos_database: str = "investigations"
    azure_cosmos_container: str = "investigation-states"

    # Azure Blob Storage
    azure_storage_connection_string: str
    azure_storage_container: str = "investigation-documents"

    # Web Search
    bing_search_api_key: str = ""
    bing_search_endpoint: str = "https://api.bing.microsoft.com/v7.0/search"

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    max_agent_iterations: int = 15

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
