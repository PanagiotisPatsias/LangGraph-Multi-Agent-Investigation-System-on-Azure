"""Recommendation Generator agent — synthesizes findings into a buy recommendation."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.config.llm_factory import create_llm
from src.graph.shopping_state import ShoppingState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Recommendation Generator agent in a multi-agent shopping system.
Your role is to compile findings from product, review, and price specialists into a concrete,
grounded buy recommendation for the user.

Recommendation Structure:
1. **Top Pick** — the single best product for the user's query, with one-paragraph rationale
2. **Runner-Up** — the second choice, with the trade-off vs the top pick
3. **Why Not the Others** — brief disqualification reasoning for remaining candidates
4. **Key Trade-offs** — table or bullet list comparing top candidates on the dimensions
   that matter (price, sound, ANC, battery, ecosystem, etc.)
5. **Deal Quality** — buy now vs wait, based on price-history signals
6. **Confidence** — overall confidence in the recommendation, with caveats

Guidelines:
- Every claim must cite the source agent (product_researcher, review_aggregator, price_comparator)
- Do not invent specs, ratings, or prices — only use what specialists reported
- If specialists disagree, acknowledge the disagreement rather than picking arbitrarily
- Be opinionated — the user wants a clear recommendation, not a buyer's guide
- Stay within the user's stated constraints (budget, must-have features)

IMPORTANT: You MUST produce a concrete Top Pick recommendation from the products that
the specialists actually reported. Do NOT refuse to recommend on the grounds that data is
incomplete — make the best call you can with what's available, and note any gaps as caveats
in the Confidence section. The user already invested time getting to this point; a hedged
recommendation with caveats is far more useful than no recommendation."""


def recommendation_generator_node(state: ShoppingState) -> dict[str, Any]:
    """LangGraph node: generate the final buy recommendation."""
    logger.info("Recommendation Generator compiling for session: %s", state["session_id"])

    llm = create_llm(temperature=0.3, max_tokens=3000)

    all_findings = []
    for f in state.get("product_findings", []):
        all_findings.append(f"### Product Research\n{f['content']}")
    for f in state.get("review_findings", []):
        all_findings.append(f"### Review Aggregation\n{f['content']}")
    for f in state.get("price_findings", []):
        all_findings.append(f"### Price Comparison\n{f['content']}")
    findings_text = "\n\n".join(all_findings) if all_findings else "No findings available."

    agent_sequence = state.get("agent_sequence", [])

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"# Recommendation Generation\n\n"
                f"**Session ID:** {state['session_id']}\n"
                f"**Query:** {state['query']}\n"
                f"**Agents Consulted:** {', '.join(agent_sequence)}\n\n"
                f"## Collected Findings\n\n{findings_text}\n\n"
                "Compile a grounded buy recommendation following the required structure. "
                "Ensure every claim cites the source agent."
            )
        ),
    ]

    response = llm.invoke(messages)
    recommendation = response.content if hasattr(response, "content") else str(response)
    sections = _extract_sections(recommendation)

    return {
        "final_recommendation": recommendation,
        "recommendation_sections": sections,
        "status": "completed",
        "messages": [
            AIMessage(content="[Recommendation Generator] Recommendation compiled.", name="recommendation_generator")
        ],
        "agent_sequence": ["recommendation_generator"],
    }


def _extract_sections(report: str) -> dict[str, str]:
    """Extract named sections from the markdown recommendation."""
    sections: dict[str, str] = {}
    current_section = "preamble"
    current_lines: list[str] = []

    for line in report.split("\n"):
        if line.startswith("## "):
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = line.lstrip("# ").strip().lower().replace(" ", "_")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections
