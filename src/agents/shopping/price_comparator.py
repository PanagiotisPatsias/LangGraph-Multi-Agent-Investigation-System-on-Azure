"""Price Comparator agent — compares current pricing and 30-day history."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config.llm_factory import create_llm
from src.graph.shopping_state import ShoppingState
from src.tools.product_search import compare_prices

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Price Comparator agent in a multi-agent shopping system.
Your role is to gather pricing data for candidate products and assess deal quality.

CRITICAL constraints:
- You MUST only call compare_prices on product names that the product_researcher already
  surfaced (listed in the message below). DO NOT invent or guess product names from your
  training data — your job is to price the candidates this session has found.

Guidelines:
- For each named candidate from the researcher, call compare_prices(product_name=...)
- Report current price, 30-day low, 30-day average, trend, best retailer
- Identify which products are currently a good deal (current < average) vs poor timing
- Flag any product where price is rising or near a recent peak
- Always cite the retailer source
- Do not invent prices — only report what the tool returns"""


def price_comparator_node(state: ShoppingState) -> dict[str, Any]:
    """LangGraph node: compare prices across retailers and time."""
    logger.info("Price Comparator processing session: %s", state["session_id"])

    llm = create_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools([compare_prices])

    prior_products = state.get("product_findings", [])
    prior_context = ""
    if prior_products:
        latest = prior_products[-1]["content"]
        prior_context = (
            "\n\n=== Candidate products surfaced by product_researcher ===\n"
            f"{latest}\n"
            "=== End of candidates — only price the products listed above ==="
        )

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Shopping query: {state['query']}{prior_context}\n\n"
                "Look up current pricing and 30-day history for each candidate. Identify which "
                "are good deals right now and which are poorly timed."
            )
        ),
    ]

    max_steps = 5
    for _ in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            if tool_call["name"] == "compare_prices":
                result = compare_prices.invoke(tool_call["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    analysis = response.content if hasattr(response, "content") else ""
    # reasoning-model fallback: see document_analyst.py for rationale.
    if not analysis.strip():
        synthesis = llm.invoke(
            messages
            + [
                HumanMessage(
                    content=(
                        "Based on the price tool outputs above, write your final pricing brief. "
                        "For each candidate priced, report current vs 30-day average, trend, and "
                        "best retailer. Identify which are good deals right now."
                    )
                )
            ]
        )
        analysis = synthesis.content if hasattr(synthesis, "content") else str(synthesis)
    findings = [
        {
            "source": "price_comparator",
            "content": analysis,
            "confidence": 0.9,
            "citations": [],
        }
    ]

    return {
        "price_findings": findings,
        "messages": [AIMessage(content=f"[Price Comparator] {analysis}", name="price_comparator")],
        "agent_sequence": ["price_comparator"],
    }
