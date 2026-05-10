"""Product Researcher agent — surfaces candidate products with specs."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config.llm_factory import create_llm
from src.graph.shopping_state import ShoppingState
from src.tools.product_search import search_products

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Product Researcher agent in a multi-agent shopping system.
Your role is to identify candidate products that match the user's query and constraints.

Guidelines:
- Use the search_products tool to surface candidates within the user's price ceiling
- Extract and summarize key specs that matter for the category (e.g., for earbuds:
  battery life, ANC, codec support, fit, water resistance)
- Note 3-5 strongest candidates with concise rationale
- Always cite the source URL for each product
- Do not invent specs — only report what the tool returns
- Rate your confidence in each candidate (high/medium/low)"""


def product_researcher_node(state: ShoppingState) -> dict[str, Any]:
    """LangGraph node: identify candidate products."""
    logger.info("Product Researcher processing session: %s", state["session_id"])

    llm = create_llm(temperature=0.1)
    llm_with_tools = llm.bind_tools([search_products])

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Shopping query: {state['query']}\n\n"
                "Search for matching products. Surface the strongest 3-5 candidates with "
                "their key specs and source URLs."
            )
        ),
    ]

    max_steps = 4
    for _ in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_products":
                result = search_products.invoke(tool_call["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    analysis = response.content if hasattr(response, "content") else ""
    # reasoning-model fallback: see document_analyst.py for rationale.
    if not analysis.strip():
        synthesis = llm.invoke(
            messages
            + [
                HumanMessage(
                    content=(
                        "Now produce your final analysis based on the tool results above. "
                        "List the candidate products you found, with their specs and source URLs. "
                        "Pick 3-5 strongest candidates with brief rationale and confidence."
                    )
                )
            ]
        )
        analysis = synthesis.content if hasattr(synthesis, "content") else str(synthesis)
    findings = [
        {
            "source": "product_researcher",
            "content": analysis,
            "confidence": 0.85,
            "citations": [],
        }
    ]

    return {
        "product_findings": findings,
        "messages": [AIMessage(content=f"[Product Researcher] {analysis}", name="product_researcher")],
        "agent_sequence": ["product_researcher"],
    }
