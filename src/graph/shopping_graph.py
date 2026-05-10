"""LangGraph shopping workflow — same architectural pattern as investigation_graph,
adapted for consumer product discovery.

Demonstrates that the multi-agent supervisor + specialists + grounded-output +
evaluator + observability pattern transfers across domains. Reuses Langfuse callback,
LLM factory, supervisor routing pattern, and evaluator (citation/hallucination
scoring applies 1:1 to product recommendations — wrong specs destroy trust).

Graph topology:
    ┌──────────────────┐
    │      START       │
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │    Supervisor    │◄─────────────────────────────┐
    └────────┬─────────┘                              │
             ▼                                        │
       ┌─────┴─────┬──────┬──────┬──────┐             │
       ▼           ▼      ▼      ▼      ▼             │
    ┌──────┐ ┌──────┐ ┌──────┐ ┌─────────────┐        │
    │Prod  │ │Review│ │Price │ │Recommendation│       │
    │Resrch│ │Aggr  │ │Cmp   │ │Generator    │        │
    └──┬───┘ └──┬───┘ └──┬───┘ └──┬──────────┘        │
       │        │       │        │                    │
       └────────┴───────┴────────┴────────────────────┘
                                  │
                             ┌────▼────┐
                             │   END   │
                             └─────────┘
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.shopping.price_comparator import price_comparator_node
from src.agents.shopping.product_researcher import product_researcher_node
from src.agents.shopping.recommendation_generator import recommendation_generator_node
from src.agents.shopping.review_aggregator import review_aggregator_node
from src.agents.shopping.supervisor import shopping_supervisor_node
from src.config.settings import get_langfuse_callback, get_settings
from src.graph.shopping_state import ShoppingState

logger = logging.getLogger(__name__)

SUPERVISOR = "shopping_supervisor"
PRODUCT_RESEARCHER = "product_researcher"
REVIEW_AGGREGATOR = "review_aggregator"
PRICE_COMPARATOR = "price_comparator"
RECOMMENDATION_GENERATOR = "recommendation_generator"


def _route_from_supervisor(state: ShoppingState) -> str:
    """Route to the next agent based on supervisor's decision."""
    next_agent = state.get("next_agent", "FINISH")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 12)

    if iteration >= max_iter:
        logger.warning("Max iterations reached — forcing recommendation generation")
        if state.get("final_recommendation"):
            return END
        return RECOMMENDATION_GENERATOR

    if next_agent == "FINISH":
        return END

    valid_agents = {
        PRODUCT_RESEARCHER,
        REVIEW_AGGREGATOR,
        PRICE_COMPARATOR,
        RECOMMENDATION_GENERATOR,
    }

    if next_agent in valid_agents:
        return next_agent

    logger.warning("Unknown agent '%s' — defaulting to recommendation_generator", next_agent)
    return RECOMMENDATION_GENERATOR


def build_shopping_graph() -> StateGraph:
    """Build and compile the shopping product-discovery graph."""
    graph = StateGraph(ShoppingState)

    graph.add_node(SUPERVISOR, shopping_supervisor_node)
    graph.add_node(PRODUCT_RESEARCHER, product_researcher_node)
    graph.add_node(REVIEW_AGGREGATOR, review_aggregator_node)
    graph.add_node(PRICE_COMPARATOR, price_comparator_node)
    graph.add_node(RECOMMENDATION_GENERATOR, recommendation_generator_node)

    graph.set_entry_point(SUPERVISOR)

    graph.add_conditional_edges(
        SUPERVISOR,
        _route_from_supervisor,
        {
            PRODUCT_RESEARCHER: PRODUCT_RESEARCHER,
            REVIEW_AGGREGATOR: REVIEW_AGGREGATOR,
            PRICE_COMPARATOR: PRICE_COMPARATOR,
            RECOMMENDATION_GENERATOR: RECOMMENDATION_GENERATOR,
            END: END,
        },
    )

    for agent in [PRODUCT_RESEARCHER, REVIEW_AGGREGATOR, PRICE_COMPARATOR, RECOMMENDATION_GENERATOR]:
        graph.add_edge(agent, SUPERVISOR)

    return graph.compile()


def create_initial_state(query: str, session_id: str | None = None) -> ShoppingState:
    """Create the initial state for a new shopping session."""
    settings = get_settings()
    sid = session_id or str(uuid.uuid4())

    return ShoppingState(
        messages=[],
        session_id=sid,
        query=query,
        status="pending",
        next_agent="",
        agent_sequence=[],
        product_findings=[],
        review_findings=[],
        price_findings=[],
        final_recommendation="",
        recommendation_sections={},
        iteration_count=0,
        max_iterations=min(12, settings.max_agent_iterations),
    )


async def run_shopping_session(
    query: str, session_id: str | None = None
) -> dict[str, Any]:
    """Run a complete shopping product-discovery workflow."""
    graph = build_shopping_graph()
    initial_state = create_initial_state(query, session_id)

    logger.info(
        "Starting shopping session %s: %s",
        initial_state["session_id"],
        query[:100],
    )

    config: dict[str, Any] = {}
    langfuse_handler = get_langfuse_callback()
    if langfuse_handler is not None:
        config["callbacks"] = [langfuse_handler]
        config["metadata"] = {
            "langfuse_session_id": initial_state["session_id"],
            "langfuse_tags": ["shopping", "langgraph"],
        }

    final_state = await graph.ainvoke(initial_state, config=config)

    logger.info(
        "Shopping session %s completed. Agents used: %s",
        final_state["session_id"],
        final_state.get("agent_sequence", []),
    )

    return dict(final_state)
