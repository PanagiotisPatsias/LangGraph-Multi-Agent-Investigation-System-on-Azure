"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "investigation-system"}


@router.get("/health/ready")
async def readiness_check() -> dict[str, str]:
    return {"status": "ready"}
