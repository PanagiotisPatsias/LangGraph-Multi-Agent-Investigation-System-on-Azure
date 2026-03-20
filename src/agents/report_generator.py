"""Report Generator agent — compiles findings into structured investigation reports."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.config.llm_factory import create_llm
from src.graph.state import GraphState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Report Generator agent in a multi-agent investigation system.
Your role is to compile findings from all specialist agents into a comprehensive,
well-structured investigation report.

Report Structure:
1. **Executive Summary** — Key findings and conclusions (2-3 paragraphs)
2. **Investigation Overview** — Scope, methodology, and data sources
3. **Document Analysis Findings** — Key insights from analyzed documents
4. **Financial Analysis Findings** — Quantitative findings, anomalies, risk metrics
5. **Open-Source Intelligence** — Web research findings and news analysis
6. **Risk Assessment** — Overall risk evaluation with severity ratings
7. **Key Findings & Red Flags** — Prioritized list of critical findings
8. **Recommendations** — Suggested next steps and actions
9. **Confidence Assessment** — Evaluation of evidence quality and limitations

Guidelines:
- Every claim must cite its source agent and evidence
- Clearly distinguish between confirmed facts and inferences
- Highlight contradictions between different sources
- Assign severity ratings: CRITICAL / HIGH / MEDIUM / LOW
- Be objective and evidence-driven — avoid speculation without evidence"""


def create_report_generator():
    """Create the LLM instance for the report generator."""
    return create_llm(temperature=0.3, max_tokens=4096)


def report_generator_node(state: GraphState) -> dict[str, Any]:
    """LangGraph node: generate the final investigation report.

    Compiles all findings from specialist agents into a structured,
    citation-grounded investigation report.
    """
    logger.info("Report Generator compiling report for: %s", state["investigation_id"])

    llm = create_report_generator()

    # Compile all findings
    all_findings = []

    for finding in state.get("document_findings", []):
        all_findings.append(f"### Document Analysis\n{finding['content']}")

    for finding in state.get("financial_findings", []):
        all_findings.append(f"### Financial Analysis\n{finding['content']}")

    for finding in state.get("web_findings", []):
        all_findings.append(f"### Web Research\n{finding['content']}")

    findings_text = "\n\n".join(all_findings) if all_findings else "No findings available."

    # Agent sequence for methodology section
    agent_sequence = state.get("agent_sequence", [])

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"# Investigation Report Generation\n\n"
                f"**Investigation ID:** {state['investigation_id']}\n"
                f"**Query:** {state['query']}\n"
                f"**Agents Consulted:** {', '.join(agent_sequence)}\n\n"
                f"## Collected Findings\n\n{findings_text}\n\n"
                "Please compile these findings into a comprehensive investigation report "
                "following the required structure. Ensure every claim is properly attributed "
                "to its source agent, and include a confidence assessment."
            )
        ),
    ]

    response = llm.invoke(messages)
    report = response.content if hasattr(response, "content") else str(response)

    # Extract sections for structured storage
    sections = _extract_sections(report)

    return {
        "final_report": report,
        "report_sections": sections,
        "status": "completed",
        "messages": [AIMessage(content=f"[Report Generator] Report compiled.", name="report_generator")],
        "agent_sequence": ["report_generator"],
    }


def _extract_sections(report: str) -> dict[str, str]:
    """Extract named sections from the markdown report."""
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
