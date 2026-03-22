"""Document Analyst agent — RAG-based document analysis."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.config.llm_factory import create_llm
from src.graph.state import GraphState
from src.tools.azure_search import search_documents

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Document Analyst agent in a multi-agent investigation system.
Your role is to analyze documents relevant to the investigation by searching through
indexed document collections and extracting key findings.

Guidelines:
- Always cite the specific document and chunk where you found information
- Structure findings as clear, evidence-backed statements
- Flag any contradictions between documents
- Identify gaps in the available documentation
- Rate your confidence in each finding (high/medium/low)

When using the search tool, craft precise queries to find the most relevant information.
If initial results are insufficient, try rephrasing or broadening your search."""


def create_document_analyst():
    """Create the LLM instance for the document analyst."""
    return create_llm(temperature=0.1)


def document_analyst_node(state: GraphState) -> dict[str, Any]:
    """LangGraph node: analyze investigation documents via RAG.

    Searches through indexed documents, extracts relevant information,
    and produces structured findings with citations.
    """
    logger.info("Document Analyst processing investigation: %s", state["investigation_id"])

    llm = create_document_analyst()
    llm_with_tools = llm.bind_tools([search_documents])

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Investigation query: {state['query']}\n\n"
                f"Investigation ID: {state['investigation_id']}\n\n"
                "Search through the available documents and extract all relevant findings. "
                "For each finding, include the source document and your confidence level. "
                "Structure your analysis with clear sections."
            )
        ),
    ]

    # Agentic loop: let the LLM call tools iteratively
    max_steps = 5
    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        # Execute tool calls
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_documents":
                args = tool_call["args"]
                # Inject investigation_id if not provided
                if "investigation_id" not in args:
                    args["investigation_id"] = state["investigation_id"]
                result = search_documents.invoke(args)
                from langchain_core.messages import ToolMessage

                messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )

    # Extract the final analysis
    analysis = response.content if hasattr(response, "content") else str(response)

    findings = [
        {
            "source": "document_analyst",
            "content": analysis,
            "confidence": 0.85,
            "citations": [],
        }
    ]

    return {
        "document_findings": findings,
        "messages": [AIMessage(content=f"[Document Analyst] {analysis}", name="document_analyst")],
        "agent_sequence": ["document_analyst"],
    }
