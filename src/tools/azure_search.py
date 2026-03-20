"""Azure AI Search tool for RAG-based document retrieval."""

from __future__ import annotations

import logging
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from langchain_core.tools import tool

from src.config.llm_factory import create_embeddings
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class AzureSearchManager:
    """Manages Azure AI Search index and provides search capabilities."""

    def __init__(self) -> None:
        settings = get_settings()
        credential = AzureKeyCredential(settings.azure_search_api_key)
        self._search_client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=credential,
        )
        self._index_client = SearchIndexClient(
            endpoint=settings.azure_search_endpoint,
            credential=credential,
        )
        self._embeddings = create_embeddings()
        self._index_name = settings.azure_search_index_name

    def ensure_index(self) -> None:
        """Create the search index if it doesn't exist."""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(
                name="investigation_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(name="document_name", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="blob_path", type=SearchFieldDataType.String),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, sortable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=3072,  # text-embedding-3-large
                vector_search_profile_name="default-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
            profiles=[
                VectorSearchProfile(
                    name="default-profile", algorithm_configuration_name="default-hnsw"
                )
            ],
        )

        index = SearchIndex(
            name=self._index_name, fields=fields, vector_search=vector_search
        )
        self._index_client.create_or_update_index(index)
        logger.info("Search index ensured: %s", self._index_name)

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Upload document chunks to the search index."""
        self._search_client.upload_documents(chunks)
        logger.info("Indexed %d chunks", len(chunks))

    def hybrid_search(
        self, query: str, investigation_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Perform hybrid (text + vector) search scoped to an investigation."""
        query_embedding = self._embeddings.embed_query(query)

        results = self._search_client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(
                    vector=query_embedding, k_nearest_neighbors=top_k, fields="embedding"
                )
            ],
            filter=f"investigation_id eq '{investigation_id}'",
            top=top_k,
            select=["id", "document_name", "content", "chunk_index"],
        )

        return [
            {
                "id": r["id"],
                "document_name": r["document_name"],
                "content": r["content"],
                "chunk_index": r["chunk_index"],
                "score": r["@search.score"],
            }
            for r in results
        ]


# Singleton for tool usage
_search_manager: AzureSearchManager | None = None


def _get_search_manager() -> AzureSearchManager:
    global _search_manager
    if _search_manager is None:
        _search_manager = AzureSearchManager()
    return _search_manager


@tool
def search_documents(query: str, investigation_id: str) -> str:
    """Search through investigation documents using hybrid (text + vector) search.

    Use this tool to find relevant information from uploaded documents.
    Returns the most relevant document chunks with source citations.

    Args:
        query: The search query describing what information to find.
        investigation_id: The ID of the investigation to search within.
    """
    manager = _get_search_manager()
    results = manager.hybrid_search(query, investigation_id)

    if not results:
        return "No relevant documents found for this query."

    formatted = []
    for r in results:
        formatted.append(
            f"[Source: {r['document_name']}, Chunk #{r['chunk_index']}] "
            f"(Score: {r['score']:.3f})\n{r['content']}"
        )

    return "\n\n---\n\n".join(formatted)
