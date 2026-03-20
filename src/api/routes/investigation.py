"""Investigation API routes."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.graph.investigation_graph import run_investigation
from src.services.cosmos_service import CosmosService
from src.services.document_service import DocumentService
from src.tools.azure_search import AzureSearchManager

logger = logging.getLogger(__name__)
router = APIRouter()


class InvestigationRequest(BaseModel):
    """Request body for starting a new investigation."""

    query: str = Field(..., min_length=10, max_length=5000, description="The investigation query")
    investigation_id: str | None = Field(
        None, description="Optional custom investigation ID"
    )


class InvestigationResponse(BaseModel):
    """Response body for investigation results."""

    investigation_id: str
    query: str
    status: str
    agent_sequence: list[str]
    final_report: str
    report_sections: dict[str, str]


@router.post("/investigations", response_model=InvestigationResponse)
async def create_investigation(request: InvestigationRequest) -> InvestigationResponse:
    """Start a new multi-agent investigation.

    The system will orchestrate multiple specialist agents to analyze the query
    and produce a comprehensive investigation report.
    """
    inv_id = request.investigation_id or str(uuid.uuid4())
    logger.info("Starting investigation %s: %s", inv_id, request.query[:100])

    try:
        result = await run_investigation(request.query, inv_id)
    except Exception as e:
        logger.error("Investigation %s failed: %s", inv_id, e)
        raise HTTPException(status_code=500, detail=f"Investigation failed: {e}")

    # Persist to Cosmos DB
    try:
        cosmos = CosmosService()
        cosmos.save_investigation(inv_id, {
            "query": request.query,
            "status": result.get("status", "completed"),
            "agent_sequence": result.get("agent_sequence", []),
            "final_report": result.get("final_report", ""),
            "report_sections": result.get("report_sections", {}),
        })
    except Exception as e:
        logger.warning("Failed to persist investigation %s: %s", inv_id, e)

    return InvestigationResponse(
        investigation_id=inv_id,
        query=request.query,
        status=result.get("status", "completed"),
        agent_sequence=result.get("agent_sequence", []),
        final_report=result.get("final_report", ""),
        report_sections=result.get("report_sections", {}),
    )


@router.get("/investigations/{investigation_id}")
async def get_investigation(investigation_id: str) -> dict[str, Any]:
    """Retrieve a completed investigation by ID."""
    cosmos = CosmosService()
    result = cosmos.get_investigation(investigation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return result


@router.get("/investigations")
async def list_investigations(limit: int = 20) -> list[dict[str, Any]]:
    """List recent investigations."""
    cosmos = CosmosService()
    return cosmos.list_investigations(limit=min(limit, 100))


@router.post("/investigations/{investigation_id}/documents")
async def upload_document(investigation_id: str, file: UploadFile) -> dict[str, Any]:
    """Upload a document for an investigation.

    The document will be processed (chunked, embedded) and indexed
    for RAG-based analysis by the Document Analyst agent.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()
    content_type = file.content_type or "application/octet-stream"

    # Process document: chunk, embed
    doc_service = DocumentService()
    chunks = doc_service.process_document(
        investigation_id, file.filename, content, content_type
    )

    # Index chunks in Azure AI Search
    search_manager = AzureSearchManager()
    search_manager.ensure_index()
    search_manager.index_chunks(chunks)

    return {
        "investigation_id": investigation_id,
        "filename": file.filename,
        "chunks_indexed": len(chunks),
        "status": "indexed",
    }


@router.delete("/investigations/{investigation_id}")
async def delete_investigation(investigation_id: str) -> dict[str, str]:
    """Delete an investigation and all associated data."""
    cosmos = CosmosService()
    cosmos.delete_investigation(investigation_id)
    return {"status": "deleted", "investigation_id": investigation_id}
