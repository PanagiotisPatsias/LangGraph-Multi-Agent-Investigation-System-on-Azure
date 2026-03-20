"""LLM factory — creates Azure OpenAI LLM and embedding instances."""

from __future__ import annotations

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from src.config.settings import get_settings


def create_llm(temperature: float = 0.1, max_tokens: int | None = None) -> AzureChatOpenAI:
    """Create an Azure OpenAI LLM instance."""
    settings = get_settings()
    kwargs: dict = {
        "azure_endpoint": settings.azure_openai_endpoint,
        "api_key": settings.azure_openai_api_key,
        "api_version": settings.azure_openai_api_version,
        "azure_deployment": settings.azure_openai_deployment,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return AzureChatOpenAI(**kwargs)


def create_embeddings() -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI embeddings instance."""
    settings = get_settings()
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
    )
