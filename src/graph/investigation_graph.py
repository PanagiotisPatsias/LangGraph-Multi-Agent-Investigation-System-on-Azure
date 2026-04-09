"""LangGraph investigation workflow — the core multi-agent orchestration graph.

Implements a supervisor-routed architecture where a supervisor agent dynamically
decides which specialist agents to invoke based on the investigation state.

Graph topology:
    ┌─────────────┐
    │   START     │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  Supervisor  │◄──────────────────────────────┐
    └──────┬──────┘                                │
           ▼                                       │
    ┌─────────────┐                                │
    │   Router    │────┬──────┬──────┬──────┐      │
    └─────────────┘    │      │      │      │      │
                       ▼      ▼      ▼      ▼      │
               ┌──────┐ ┌────┐ ┌────┐ ┌──────┐    │
               │ Doc  │ │Fin │ │Web │ │Report│    │
               │Analyst│ │Anl │ │Res │ │ Gen  │    │
               └──┬───┘ └─┬──┘ └─┬──┘ └──┬───┘    │
                  │       │      │       │         │
                  └───────┴──────┴───────┴─────────┘
                                         │
                                    ┌────▼────┐
                                    │  END    │
                                    └─────────┘
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.document_analyst import document_analyst_node
from src.agents.financial_analyst import financial_analyst_node
from src.agents.report_generator import report_generator_node
from src.agents.supervisor import supervisor_node
from src.agents.web_researcher import web_researcher_node
from src.config.settings import get_settings
from src.graph.state import GraphState

logger = logging.getLogger(__name__)

# Agent node names
SUPERVISOR = "supervisor"
DOCUMENT_ANALYST = "document_analyst"
FINANCIAL_ANALYST = "financial_analyst"
WEB_RESEARCHER = "web_researcher"
REPORT_GENERATOR = "report_generator"


def _route_from_supervisor(state: GraphState) -> str:
    """Route to the next agent based on supervisor's decision."""
    next_agent = state.get("next_agent", "FINISH")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 15 )

    # Safety: prevent infinite loops
    if iteration >= max_iter:
        logger.warning("Max iterations reached — forcing report generation")
        if state.get("final_report"):
            return END
        return REPORT_GENERATOR

    if next_agent == "FINISH":
        return END

    valid_agents = {
        DOCUMENT_ANALYST,
        FINANCIAL_ANALYST,
        WEB_RESEARCHER,
        REPORT_GENERATOR,
    }

    if next_agent in valid_agents:
        return next_agent

    logger.warning("Unknown agent '%s' — defaulting to report_generator", next_agent)
    return REPORT_GENERATOR


def build_investigation_graph() -> StateGraph:
    """Build and compile the multi-agent investigation graph.

    Returns a compiled LangGraph that orchestrates the investigation workflow
    with dynamic supervisor-based routing.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node(SUPERVISOR, supervisor_node)
    graph.add_node(DOCUMENT_ANALYST, document_analyst_node)
    graph.add_node(FINANCIAL_ANALYST, financial_analyst_node)
    graph.add_node(WEB_RESEARCHER, web_researcher_node)
    graph.add_node(REPORT_GENERATOR, report_generator_node)

    # Entry point
    graph.set_entry_point(SUPERVISOR)

    # Conditional routing from supervisor
    graph.add_conditional_edges(
        SUPERVISOR,
        _route_from_supervisor,
        {
            DOCUMENT_ANALYST: DOCUMENT_ANALYST,
            FINANCIAL_ANALYST: FINANCIAL_ANALYST,
            WEB_RESEARCHER: WEB_RESEARCHER,
            REPORT_GENERATOR: REPORT_GENERATOR,
            END: END,
        },
    )

    # All specialist agents route back to supervisor
    for agent in [DOCUMENT_ANALYST, FINANCIAL_ANALYST, WEB_RESEARCHER, REPORT_GENERATOR]:
        graph.add_edge(agent, SUPERVISOR)

    return graph.compile()


def create_initial_state(query: str, investigation_id: str | None = None) -> GraphState:
    """Create the initial state for a new investigation."""
    settings = get_settings()
    inv_id = investigation_id or str(uuid.uuid4())

    return GraphState(
        messages=[],
        investigation_id=inv_id,
        query=query,
        status="pending",
        next_agent="",
        agent_sequence=[],
        document_findings=[],
        financial_findings=[],
        web_findings=[],
        final_report="",
        report_sections={},
        iteration_count=0,
        max_iterations=settings.max_agent_iterations,
    )


async def run_investigation(
    query: str, investigation_id: str | None = None
) -> dict[str, Any]:
    """Run a complete investigation workflow.

    Args:
        query: The investigation query/prompt.
        investigation_id: Optional custom investigation ID.

    Returns:
        The final investigation state including the report.
    """
    graph = build_investigation_graph()
    initial_state = create_initial_state(query, investigation_id)

    logger.info(
        "Starting investigation %s: %s",
        initial_state["investigation_id"],
        query[:100],
    )

    # Execute the graph
    final_state = await graph.ainvoke(initial_state)

    logger.info(
        "Investigation %s completed. Agents used: %s",
        final_state["investigation_id"],
        final_state.get("agent_sequence", []),
    )

    return dict(final_state)
