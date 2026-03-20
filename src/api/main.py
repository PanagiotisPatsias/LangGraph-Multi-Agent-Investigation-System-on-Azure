"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.health import router as health_router
from src.api.routes.investigation import router as investigation_router
from src.config.settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

app = FastAPI(
    title="Multi-Agent Investigation System",
    description=(
        "AI-powered financial investigation system using LangGraph multi-agent "
        "orchestration on Azure. Analyzes documents, financial data, and web "
        "intelligence to produce structured investigation reports."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(health_router, tags=["health"])
app.include_router(investigation_router, prefix="/api/v1", tags=["investigations"])
