"""Tests for the LangGraph investigation graph."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.graph.investigation_graph import (
    _route_from_supervisor,
    build_investigation_graph,
    create_initial_state,
)
from src.graph.state import GraphState


class TestRouting:
    """Test the supervisor routing logic."""

    def test_route_to_document_analyst(self, sample_state: GraphState):
        sample_state["next_agent"] = "document_analyst"
        assert _route_from_supervisor(sample_state) == "document_analyst"

    def test_route_to_financial_analyst(self, sample_state: GraphState):
        sample_state["next_agent"] = "financial_analyst"
        assert _route_from_supervisor(sample_state) == "financial_analyst"

    def test_route_to_web_researcher(self, sample_state: GraphState):
        sample_state["next_agent"] = "web_researcher"
        assert _route_from_supervisor(sample_state) == "web_researcher"

    def test_route_to_report_generator(self, sample_state: GraphState):
        sample_state["next_agent"] = "report_generator"
        assert _route_from_supervisor(sample_state) == "report_generator"

    def test_route_finish(self, sample_state: GraphState):
        sample_state["next_agent"] = "FINISH"
        result = _route_from_supervisor(sample_state)
        assert result == "__end__"

    def test_route_unknown_defaults_to_report(self, sample_state: GraphState):
        sample_state["next_agent"] = "nonexistent_agent"
        assert _route_from_supervisor(sample_state) == "report_generator"

    def test_max_iterations_forces_report(self, sample_state: GraphState):
        sample_state["iteration_count"] = 15
        sample_state["max_iterations"] = 15
        sample_state["next_agent"] = "document_analyst"
        assert _route_from_supervisor(sample_state) == "report_generator"

    def test_max_iterations_with_report_ends(self, sample_state: GraphState):
        sample_state["iteration_count"] = 15
        sample_state["max_iterations"] = 15
        sample_state["final_report"] = "Some report"
        result = _route_from_supervisor(sample_state)
        assert result == "__end__"


class TestInitialState:
    """Test initial state creation."""

    def test_creates_valid_state(self):
        state = create_initial_state("Investigate XYZ Corp")
        assert state["query"] == "Investigate XYZ Corp"
        assert state["status"] == "pending"
        assert state["iteration_count"] == 0
        assert len(state["investigation_id"]) > 0

    def test_custom_investigation_id(self):
        state = create_initial_state("Test query", investigation_id="custom-id-123")
        assert state["investigation_id"] == "custom-id-123"

    def test_empty_findings(self):
        state = create_initial_state("Test")
        assert state["document_findings"] == []
        assert state["financial_findings"] == []
        assert state["web_findings"] == []


class TestGraphBuild:
    """Test that the graph compiles correctly."""

    def test_graph_compiles(self):
        graph = build_investigation_graph()
        assert graph is not None
