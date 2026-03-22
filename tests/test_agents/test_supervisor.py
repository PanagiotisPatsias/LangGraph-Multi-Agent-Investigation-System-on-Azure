"""Tests for the Supervisor agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.supervisor import SupervisorDecision, supervisor_node
from src.graph.state import GraphState


class TestSupervisorDecision:
    """Test the supervisor's routing logic."""

    def test_supervisor_routes_to_agent(self, sample_state: GraphState, mock_llm: MagicMock):
        """Supervisor should route to a valid agent."""
        decision = SupervisorDecision(
            next_agent="document_analyst",
            reasoning="Query involves document analysis",
        )
        mock_llm.invoke.return_value = decision

        with patch("src.agents.supervisor.create_supervisor", return_value=mock_llm):
            result = supervisor_node(sample_state)

        assert result["next_agent"] == "document_analyst"
        assert "supervisor" in result["agent_sequence"]
        assert result["iteration_count"] == 1

    def test_supervisor_routes_to_finish(
        self, sample_state_with_findings: GraphState, mock_llm: MagicMock
    ):
        """Supervisor should route to FINISH when investigation is complete."""
        decision = SupervisorDecision(
            next_agent="FINISH",
            reasoning="Report is complete",
        )
        mock_llm.invoke.return_value = decision

        with patch("src.agents.supervisor.create_supervisor", return_value=mock_llm):
            result = supervisor_node(sample_state_with_findings)

        assert result["next_agent"] == "FINISH"

    def test_supervisor_increments_iteration(
        self, sample_state: GraphState, mock_llm: MagicMock
    ):
        """Each supervisor call should increment the iteration counter."""
        decision = SupervisorDecision(
            next_agent="web_researcher",
            reasoning="Need web intelligence",
        )
        mock_llm.invoke.return_value = decision

        with patch("src.agents.supervisor.create_supervisor", return_value=mock_llm):
            result = supervisor_node(sample_state)

        assert result["iteration_count"] == sample_state["iteration_count"] + 1


class TestSupervisorDecisionModel:
    """Test the Pydantic decision model."""

    def test_valid_agents(self):
        for agent in ["document_analyst", "financial_analyst", "web_researcher", "report_generator", "FINISH"]:
            d = SupervisorDecision(next_agent=agent, reasoning="test")
            assert d.next_agent == agent

    def test_invalid_agent_rejected(self):
        with pytest.raises(Exception):
            SupervisorDecision(next_agent="invalid_agent", reasoning="test")
