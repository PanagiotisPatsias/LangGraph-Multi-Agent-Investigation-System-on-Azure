"""Tests for investigation API routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthRoutes:
    """Test health check endpoints."""

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_check(self):
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestInvestigationRoutes:
    """Test investigation CRUD endpoints."""

    @patch("src.api.routes.investigation.run_investigation")
    @patch("src.api.routes.investigation.CosmosService")
    def test_create_investigation(self, mock_cosmos_cls, mock_run):
        mock_run.return_value = {
            "investigation_id": "test-123",
            "status": "completed",
            "agent_sequence": ["supervisor", "document_analyst", "report_generator"],
            "final_report": "# Investigation Report\n\nTest report content.",
            "report_sections": {"executive_summary": "Test summary"},
        }
        mock_cosmos = MagicMock()
        mock_cosmos_cls.return_value = mock_cosmos

        response = client.post(
            "/api/v1/investigations",
            json={"query": "Investigate suspicious trading in XYZ Corp stock over last quarter"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "final_report" in data

    def test_create_investigation_short_query(self):
        response = client.post(
            "/api/v1/investigations",
            json={"query": "short"},
        )
        assert response.status_code == 422  # Validation error

    @patch("src.api.routes.investigation.CosmosService")
    def test_get_investigation_not_found(self, mock_cosmos_cls):
        mock_cosmos = MagicMock()
        mock_cosmos.get_investigation.return_value = None
        mock_cosmos_cls.return_value = mock_cosmos

        response = client.get("/api/v1/investigations/nonexistent-id")
        assert response.status_code == 404

    @patch("src.api.routes.investigation.CosmosService")
    def test_list_investigations(self, mock_cosmos_cls):
        mock_cosmos = MagicMock()
        mock_cosmos.list_investigations.return_value = [
            {"id": "inv-1", "query": "Test 1"},
            {"id": "inv-2", "query": "Test 2"},
        ]
        mock_cosmos_cls.return_value = mock_cosmos

        response = client.get("/api/v1/investigations")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
