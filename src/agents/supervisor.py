"""Supervisor agent — orchestrates the multi-agent investigation workflow."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.config.llm_factory import create_llm
from src.graph.state import GraphState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Supervisor agent orchestrating a multi-agent financial investigation.
You coordinate specialist agents to investigate a query thoroughly.

Available agents:
- **document_analyst**: Searches and analyzes uploaded investigation documents (RAG)
- **financial_analyst**: Performs quantitative financial analysis and anomaly detection
- **web_researcher**: Gathers intelligence from web sources and news
- **report_generator**: Compiles all findings into a final structured report

Your job is to:
1. Analyze the investigation query
2. Decide which agents to invoke and in what order
3. After agents have gathered enough evidence, route to report_generator
4. Route to FINISH only after the report is generated

Decision guidelines:
- For document-heavy queries → start with document_analyst
- For financial data queries → prioritize financial_analyst  
- For entity/company investigations → start with web_researcher
- For comprehensive investigations → use all three analysts, then report
- Always route to report_generator before finishing
- Avoid calling the same agent more than twice unless necessary
- Consider iteration count — wrap up if approaching the limit"""


class SupervisorDecision(BaseModel):
    """Structured output for supervisor routing decisions."""

    next_agent: Literal[
        "document_analyst",
        "financial_analyst",
        "web_researcher",
        "report_generator",
        "FINISH",
    ] = Field(description="The next agent to route to")
    reasoning: str = Field(description="Brief explanation of the routing decision")


def create_supervisor():
    """Create the LLM instance for the supervisor."""
    return create_llm(temperature=0.0)


def supervisor_node(state: GraphState) -> dict[str, Any]:
    """LangGraph node: supervisor decides which agent to invoke next.

    Analyzes the current investigation state and routes to the most
    appropriate specialist agent or terminates the workflow.
    """
    logger.info(
        "Supervisor routing for investigation: %s (iteration %d)",
        state["investigation_id"],
        state.get("iteration_count", 0),
    )

    llm = create_supervisor()
    structured_llm = llm.with_structured_output(SupervisorDecision)

    # Build context summary
    agent_history = state.get("agent_sequence", [])
    doc_count = len(state.get("document_findings", []))
    fin_count = len(state.get("financial_findings", []))
    web_count = len(state.get("web_findings", []))
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 15)
    has_report = bool(state.get("final_report"))

    context = (
        f"Investigation query: {state['query']}\n"
        f"Iteration: {iteration}/{max_iter}\n"
        f"Agents already invoked: {agent_history}\n"
        f"Document findings: {doc_count}\n"
        f"Financial findings: {fin_count}\n"
        f"Web findings: {web_count}\n"
        f"Report generated: {has_report}\n"
    )

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Current investigation state:\n\n{context}\n\nDecide the next step."),
    ]

    decision: SupervisorDecision = structured_llm.invoke(messages)  # type: ignore[assignment]
    logger.info("Supervisor decision: %s — %s", decision.next_agent, decision.reasoning)

    return {
        "next_agent": decision.next_agent,
        "iteration_count": iteration + 1,
        "messages": [
            AIMessage(
                content=f"[Supervisor] Routing to {decision.next_agent}: {decision.reasoning}",
                name="supervisor",
            )
        ],
        "agent_sequence": ["supervisor"],
    }
