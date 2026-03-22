"""Investigation graph state definition.

Defines the shared state that flows through all agents in the LangGraph
investigation pipeline. Uses annotated reducers for proper message merging.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


@dataclass
class Finding:
    """A single finding produced by an agent."""

    source: str
    content: str
    confidence: float
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _merge_findings(left: list[Finding], right: list[Finding]) -> list[Finding]:
    return left + right


class InvestigationState:
    """Typed state for the multi-agent investigation graph.

    Uses TypedDict-style annotations for LangGraph compatibility while
    providing clear schema for all agent interactions.
    """

    pass


# LangGraph requires TypedDict for state — we use annotations for reducers
from typing import TypedDict  # noqa: E402


class GraphState(TypedDict):
    """Core state flowing through the investigation graph."""

    # Message history with automatic merging
    messages: Annotated[list[BaseMessage], add_messages]

    # Investigation metadata
    investigation_id: str
    query: str
    status: str  # "pending" | "in_progress" | "completed" | "failed"

    # Agent routing
    next_agent: str
    agent_sequence: Annotated[list[str], operator.add]

    # Findings from each specialist agent
    document_findings: Annotated[list[dict[str, Any]], operator.add]
    financial_findings: Annotated[list[dict[str, Any]], operator.add]
    web_findings: Annotated[list[dict[str, Any]], operator.add]

    # Final output
    final_report: str
    report_sections: dict[str, str]

    # Control
    iteration_count: int
    max_iterations: int
