"""Tests for the Document Analyst agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agents.document_analyst import document_analyst_node
from src.graph.state import GraphState


class TestDocumentAnalyst:
    """Test the document analyst agent node."""

    def test_produces_findings(self, sample_state: GraphState, mock_llm: MagicMock):
        """Document analyst should produce document findings."""
        with patch("src.agents.document_analyst.create_document_analyst", return_value=mock_llm):
            result = document_analyst_node(sample_state)

        assert "document_findings" in result
        assert len(result["document_findings"]) > 0
        assert result["document_findings"][0]["source"] == "document_analyst"

    def test_adds_to_agent_sequence(self, sample_state: GraphState, mock_llm: MagicMock):
        """Document analyst should record itself in agent sequence."""
        with patch("src.agents.document_analyst.create_document_analyst", return_value=mock_llm):
            result = document_analyst_node(sample_state)

        assert "document_analyst" in result["agent_sequence"]

    def test_includes_messages(self, sample_state: GraphState, mock_llm: MagicMock):
        """Document analyst should add messages to state."""
        with patch("src.agents.document_analyst.create_document_analyst", return_value=mock_llm):
            result = document_analyst_node(sample_state)

        assert len(result["messages"]) > 0
        assert "[Document Analyst]" in result["messages"][0].content
