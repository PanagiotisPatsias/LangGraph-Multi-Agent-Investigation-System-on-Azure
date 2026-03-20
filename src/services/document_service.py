"""Document processing service — chunking, embedding, and indexing."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.llm_factory import create_embeddings
from src.config.settings import get_settings
from src.services.blob_service import BlobService

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200


class DocumentService:
    """Processes documents: extract text, chunk, embed, and prepare for indexing."""

    def __init__(self, blob_service: BlobService | None = None) -> None:
        self._blob = blob_service or BlobService()
        self._embeddings = create_embeddings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process_document(
        self, investigation_id: str, filename: str, content: bytes, content_type: str
    ) -> list[dict[str, Any]]:
        """Process a document: upload, chunk, embed, return indexed chunks."""
        # Upload to blob
        blob_path = self._blob.upload_document(investigation_id, filename, content, content_type)

        # Extract text (supports plain text and basic PDF)
        text = self._extract_text(content, content_type)

        # Chunk
        chunks = self._splitter.split_text(text)
        logger.info("Split document %s into %d chunks", filename, len(chunks))

        # Embed
        embeddings = self._embeddings.embed_documents(chunks)

        # Build index-ready documents
        indexed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.sha256(f"{blob_path}:{i}".encode()).hexdigest()[:16]
            indexed_chunks.append(
                {
                    "id": chunk_id,
                    "investigation_id": investigation_id,
                    "document_name": filename,
                    "blob_path": blob_path,
                    "chunk_index": i,
                    "content": chunk,
                    "embedding": embedding,
                }
            )

        return indexed_chunks

    @staticmethod
    def _extract_text(content: bytes, content_type: str) -> str:
        """Extract text from document bytes based on content type."""
        if content_type in ("text/plain", "text/markdown", "text/csv"):
            return content.decode("utf-8", errors="replace")

        if content_type == "application/pdf":
            return _extract_pdf_text(content)

        # Fallback: try decoding as text
        return content.decode("utf-8", errors="replace")


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes using a lightweight approach."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("PyMuPDF not installed — falling back to raw text extraction")
        return content.decode("utf-8", errors="replace")
