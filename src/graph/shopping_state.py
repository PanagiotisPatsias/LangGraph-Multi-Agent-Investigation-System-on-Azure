"""Shopping graph state — parallel to investigation state, domain-renamed fields."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class ShoppingState(TypedDict):
    """State flowing through the shopping product-discovery graph."""

    messages: Annotated[list[BaseMessage], add_messages]

    session_id: str
    query: str
    status: str  # "pending" | "in_progress" | "completed" | "failed"

    next_agent: str
    agent_sequence: Annotated[list[str], operator.add]

    product_findings: Annotated[list[dict[str, Any]], operator.add]
    review_findings: Annotated[list[dict[str, Any]], operator.add]
    price_findings: Annotated[list[dict[str, Any]], operator.add]

    final_recommendation: str
    recommendation_sections: dict[str, str]

    iteration_count: int
    max_iterations: int
