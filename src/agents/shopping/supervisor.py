"""Shopping supervisor — same routing pattern as the investigation supervisor,
adapted for product-discovery decisions."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.config.llm_factory import create_llm
from src.graph.shopping_state import ShoppingState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Supervisor agent orchestrating a multi-agent product-discovery workflow.
You coordinate specialist agents to help a user decide what to buy.

Available agents:
- **product_researcher**: Searches catalogs and surfaces candidate products with specs
- **review_aggregator**: Fetches and summarizes professional + user reviews
- **price_comparator**: Compares current prices across retailers and price history
- **recommendation_generator**: Synthesizes findings into a grounded buy recommendation

Your job is to:
1. Analyze the shopping query (what is the user trying to buy and under what constraints?)
2. Decide which agents to invoke and in what order
3. After agents have gathered enough evidence, route to recommendation_generator
4. Route to FINISH only after the recommendation is generated

Decision guidelines:
- For unfamiliar product categories → start with product_researcher to identify candidates
- For decision-stage queries with named products → fetch reviews and prices in parallel-ish order
- For price-sensitive queries → prioritize price_comparator early
- Always route to recommendation_generator before finishing
- Avoid calling the same agent more than twice unless necessary
- Wrap up if approaching the iteration limit"""


class ShoppingSupervisorDecision(BaseModel):
    """Structured output for shopping supervisor routing decisions."""

    next_agent: Literal[
        "product_researcher",
        "review_aggregator",
        "price_comparator",
        "recommendation_generator",
        "FINISH",
    ] = Field(description="The next agent to route to")
    reasoning: str = Field(description="Brief explanation of the routing decision")


def shopping_supervisor_node(state: ShoppingState) -> dict[str, Any]:
    """LangGraph node: shopping supervisor decides which agent to invoke next."""
    logger.info(
        "Shopping supervisor routing for session: %s (iteration %d)",
        state["session_id"],
        state.get("iteration_count", 0),
    )

    llm = create_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ShoppingSupervisorDecision)

    agent_history = state.get("agent_sequence", [])
    prod_count = len(state.get("product_findings", []))
    rev_count = len(state.get("review_findings", []))
    price_count = len(state.get("price_findings", []))
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 12)
    has_recommendation = bool(state.get("final_recommendation"))

    context = (
        f"Shopping query: {state['query']}\n"
        f"Iteration: {iteration}/{max_iter}\n"
        f"Agents already invoked: {agent_history}\n"
        f"Product findings: {prod_count}\n"
        f"Review findings: {rev_count}\n"
        f"Price findings: {price_count}\n"
        f"Recommendation generated: {has_recommendation}\n"
    )

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Current session state:\n\n{context}\n\nDecide the next step."),
    ]

    decision: ShoppingSupervisorDecision = structured_llm.invoke(messages)  # type: ignore[assignment]
    logger.info("Shopping supervisor decision: %s — %s", decision.next_agent, decision.reasoning)

    return {
        "next_agent": decision.next_agent,
        "iteration_count": iteration + 1,
        "messages": [
            AIMessage(
                content=f"[Shopping Supervisor] Routing to {decision.next_agent}: {decision.reasoning}",
                name="shopping_supervisor",
            )
        ],
        "agent_sequence": ["shopping_supervisor"],
    }
