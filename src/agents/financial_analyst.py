"""Financial Analyst agent — quantitative financial analysis and anomaly detection."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.config.llm_factory import create_llm
from src.graph.state import GraphState
from src.tools.financial_data import analyze_financial_data, compute_risk_metrics

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Financial Analyst agent in a multi-agent investigation system.
Your role is to analyze financial data, detect anomalies, assess risk, and identify
suspicious patterns relevant to the investigation.

Guidelines:
- Use quantitative tools to analyze financial time series data
- Look for anomalous patterns: unusual spikes, sudden drops, abnormal volumes
- Compute risk metrics when relevant (Sharpe ratio, VaR, drawdowns)
- Contextualize findings with financial domain knowledge
- Identify potential red flags: market manipulation, unusual trading patterns, etc.
- Provide clear numerical evidence to support your conclusions
- Rate your confidence in each finding (high/medium/low)"""


def create_financial_analyst():
    """Create the LLM instance for the financial analyst."""
    return create_llm(temperature=0.1)


def financial_analyst_node(state: GraphState) -> dict[str, Any]:
    """LangGraph node: perform financial analysis on investigation data.

    Analyzes financial indicators, detects anomalies, and produces
    quantitative findings with risk assessments.
    """
    logger.info("Financial Analyst processing investigation: %s", state["investigation_id"])

    llm = create_financial_analyst()
    llm_with_tools = llm.bind_tools([analyze_financial_data, compute_risk_metrics])

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Investigation query: {state['query']}\n\n"
                "Analyze any financial data relevant to this investigation. "
                "Look for anomalies, compute risk metrics where applicable, "
                "and identify any suspicious financial patterns. "
                "Provide a structured financial analysis with quantitative evidence."
            )
        ),
    ]

    # Include any document findings as context
    if state.get("document_findings"):
        doc_context = "\n".join(
            f"- {f['content'][:500]}" for f in state["document_findings"]
        )
        messages.append(
            HumanMessage(
                content=f"Context from document analysis:\n{doc_context}"
            )
        )

    # Agentic loop
    max_steps = 5
    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_fn = {
                "analyze_financial_data": analyze_financial_data,
                "compute_risk_metrics": compute_risk_metrics,
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
            "source": "financial_analyst",
            "content": analysis,
            "confidence": 0.8,
            "citations": [],
        }
    ]

    return {
        "financial_findings": findings,
        "messages": [
            AIMessage(content=f"[Financial Analyst] {analysis}", name="financial_analyst")
        ],
        "agent_sequence": ["financial_analyst"],
    }
