"""Review Aggregator agent — fetches and summarizes professional reviews."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config.llm_factory import create_llm
from src.graph.shopping_state import ShoppingState
from src.tools.product_search import fetch_reviews

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Review Aggregator agent in a multi-agent shopping system.
Your role is to gather professional reviews for candidate products and synthesize them.

CRITICAL constraints:
- You MUST only call fetch_reviews on product names that the product_researcher already
  surfaced (listed in the message below). DO NOT invent or guess product names from your
  training data — your job is to aggregate reviews for THIS session's candidates only.
- If the upstream researcher surfaced no candidates, say so plainly and stop.

Guidelines:
- For each named candidate from the researcher, call fetch_reviews(product_name=...)
- Extract reviewer ratings, key praise points, and key criticisms
- Cross-reference across reviewers — flag agreement and disagreement
- Always cite the reviewer source and URL from the tool output
- Do not fabricate ratings or quotes — only report what the tool returns"""


def review_aggregator_node(state: ShoppingState) -> dict[str, Any]:
    """LangGraph node: aggregate professional reviews for candidate products."""
    logger.info("Review Aggregator processing session: %s", state["session_id"])

    llm = create_llm(temperature=0.1)
    llm_with_tools = llm.bind_tools([fetch_reviews])

    prior_products = state.get("product_findings", [])
    prior_context = ""
    if prior_products:
        latest = prior_products[-1]["content"]
        prior_context = (
            "\n\n=== Candidate products surfaced by product_researcher ===\n"
            f"{latest}\n"
            "=== End of candidates — only fetch reviews for products listed above ==="
        )

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Shopping query: {state['query']}{prior_context}\n\n"
                "Fetch reviews for each named candidate product and synthesize a concise "
                "review brief. Highlight consensus and disagreement across reviewers."
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
            if tool_call["name"] == "fetch_reviews":
                result = fetch_reviews.invoke(tool_call["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    analysis = response.content if hasattr(response, "content") else ""
    # reasoning-model fallback: see document_analyst.py for rationale.
    if not analysis.strip():
        synthesis = llm.invoke(
            messages
            + [
                HumanMessage(
                    content=(
                        "Based on the review tool outputs above, write your final aggregated "
                        "review brief. For each candidate covered, summarize ratings, key praise, "
                        "and key criticisms with reviewer source citations."
                    )
                )
            ]
        )
        analysis = synthesis.content if hasattr(synthesis, "content") else str(synthesis)
    findings = [
        {
            "source": "review_aggregator",
            "content": analysis,
            "confidence": 0.8,
            "citations": [],
        }
    ]

    return {
        "review_findings": findings,
        "messages": [AIMessage(content=f"[Review Aggregator] {analysis}", name="review_aggregator")],
        "agent_sequence": ["review_aggregator"],
    }
