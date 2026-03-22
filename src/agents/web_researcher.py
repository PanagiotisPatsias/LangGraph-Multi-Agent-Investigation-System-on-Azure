"""Web Researcher agent — gathers intelligence from web sources."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.config.llm_factory import create_llm
from src.graph.state import GraphState
from src.tools.web_search import web_search, web_search_news

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Web Researcher agent in a multi-agent investigation system.
Your role is to gather intelligence from web sources to support the investigation.

Guidelines:
- Search for relevant public information about entities under investigation
- Look for news articles, regulatory filings, public records
- Cross-reference information from multiple sources
- Flag potential misinformation or unverified claims
- Summarize key findings with source URLs for verification
- Focus on recent and relevant information
- Rate your confidence in each finding (high/medium/low)

Use both general web search and news search to get comprehensive coverage."""


def create_web_researcher():
    """Create the LLM instance for the web researcher."""
    return create_llm(temperature=0.2)


def web_researcher_node(state: GraphState) -> dict[str, Any]:
    """LangGraph node: research web sources for investigation intelligence.

    Searches the web for relevant information, news, and public records
    related to the investigation query.
    """
    logger.info("Web Researcher processing investigation: %s", state["investigation_id"])

    llm = create_web_researcher()
    llm_with_tools = llm.bind_tools([web_search, web_search_news])

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Investigation query: {state['query']}\n\n"
                "Search the web for relevant information about the entities and topics "
                "in this investigation. Use both general search and news search. "
                "Compile a comprehensive intelligence brief with source attribution."
            )
        ),
    ]

    # Agentic loop
    max_steps = 6
    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_fn = {
                "web_search": web_search,
                "web_search_news": web_search_news,
            }.get(tool_call["name"])

            if tool_fn:
                result = tool_fn.invoke(tool_call["args"])
                from langchain_core.messages import ToolMessage

                messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )

    analysis = response.content if hasattr(response, "content") else str(response)

    findings = [
        {
            "source": "web_researcher",
            "content": analysis,
            "confidence": 0.7,
            "citations": [],
        }
    ]

    return {
        "web_findings": findings,
        "messages": [AIMessage(content=f"[Web Researcher] {analysis}", name="web_researcher")],
        "agent_sequence": ["web_researcher"],
    }
