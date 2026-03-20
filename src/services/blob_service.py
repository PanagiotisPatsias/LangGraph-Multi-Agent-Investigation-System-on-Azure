"""Azure Blob Storage service for document management."""

from __future__ import annotations

import logging
from io import BytesIO

from azure.storage.blob import BlobServiceClient, ContentSettings

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class BlobService:
    """Handles document upload, download, and listing in Azure Blob Storage."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
        self._container_name = settings.azure_storage_container
        self._ensure_container()

    def _ensure_container(self) -> None:
        container = self._client.get_container_client(self._container_name)
        if not container.exists():
            self._client.create_container(self._container_name)
            logger.info("Created blob container: %s", self._container_name)

    def upload_document(
        self, investigation_id: str, filename: str, content: bytes, content_type: str
    ) -> str:
        """Upload a document and return the blob path."""
        blob_path = f"{investigation_id}/{filename}"
        blob_client = self._client.get_blob_client(self._container_name, blob_path)
        blob_client.upload_blob(
            content,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        logger.info("Uploaded document: %s", blob_path)
        return blob_path

    def download_document(self, blob_path: str) -> bytes:
        """Download a document by blob path."""
        blob_client = self._client.get_blob_client(self._container_name, blob_path)
        stream = BytesIO()
        blob_client.download_blob().readinto(stream)
        stream.seek(0)
        return stream.read()

    def list_documents(self, investigation_id: str) -> list[str]:
        """List all documents for an investigation."""
        container = self._client.get_container_client(self._container_name)
        prefix = f"{investigation_id}/"
        return [blob.name for blob in container.list_blobs(name_starts_with=prefix)]

    def delete_document(self, blob_path: str) -> None:
        """Delete a document from blob storage."""
        blob_client = self._client.get_blob_client(self._container_name, blob_path)
        blob_client.delete_blob()
        logger.info("Deleted document: %s", blob_path)
