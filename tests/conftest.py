"""Shared test fixtures and configuration."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Set test environment variables before importing settings
os.environ.update({
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "test-key",
    "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-large",
    "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net",
    "AZURE_SEARCH_API_KEY": "test-search-key",
    "AZURE_SEARCH_INDEX_NAME": "test-index",
    "AZURE_COSMOS_ENDPOINT": "https://test.documents.azure.com:443/",
    "AZURE_COSMOS_KEY": "test-cosmos-key",
    "AZURE_COSMOS_DATABASE": "test-db",
    "AZURE_COSMOS_CONTAINER": "test-container",
    "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=test",
    "AZURE_STORAGE_CONTAINER": "test-docs",
    "BING_SEARCH_API_KEY": "test-bing-key",
    "APP_ENV": "test",
})

from src.graph.state import GraphState  # noqa: E402


@pytest.fixture
def sample_state() -> GraphState:
    """Create a sample investigation state for testing."""
    return GraphState(
        messages=[],
        investigation_id="test-inv-001",
        query="Investigate unusual trading patterns in XYZ Corp stock",
        status="pending",
        next_agent="",
        agent_sequence=[],
        document_findings=[],
        financial_findings=[],
        web_findings=[],
        final_report="",
        report_sections={},
        iteration_count=0,
        max_iterations=15,
    )


@pytest.fixture
def sample_state_with_findings() -> GraphState:
    """Create a state with pre-existing findings."""
    return GraphState(
        messages=[],
        investigation_id="test-inv-002",
        query="Investigate potential market manipulation in ABC Ltd",
        status="in_progress",
        next_agent="",
        agent_sequence=["supervisor", "document_analyst", "supervisor", "financial_analyst"],
        document_findings=[
            {
                "source": "document_analyst",
                "content": "Found suspicious trading volume in quarterly report Q3 2025.",
                "confidence": 0.85,
                "citations": ["quarterly_report_q3.pdf"],
            }
        ],
        financial_findings=[
            {
                "source": "financial_analyst",
                "content": "Detected 3 anomalous volume spikes with z-scores > 3.0.",
                "confidence": 0.9,
                "citations": [],
            }
        ],
        web_findings=[],
        final_report="",
        report_sections={},
        iteration_count=4,
        max_iterations=15,
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing agents without API calls."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Mock analysis result"
    mock_response.tool_calls = []
    mock.invoke.return_value = mock_response
    mock.bind_tools.return_value = mock
    mock.with_structured_output.return_value = mock
    return mock
