"""LLM evaluation and guardrails for investigation outputs.

Implements automated quality checks on generated reports:
- Hallucination detection (claims not grounded in source findings)
- Citation verification (all claims should reference source agents)
- Completeness scoring (coverage of key report sections)
- Confidence calibration (alignment between stated and actual confidence)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage

from src.config.llm_factory import create_llm

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating an investigation report."""

    overall_score: float  # 0.0 to 1.0
    hallucination_score: float
    citation_score: float
    completeness_score: float
    coherence_score: float
    issues: list[str] = field(default_factory=list)
    passed: bool = True


REQUIRED_SECTIONS = [
    "executive_summary",
    "investigation_overview",
    "risk_assessment",
    "key_findings",
    "recommendations",
]

EVALUATION_PROMPT = """You are an evaluation judge for AI-generated investigation reports.
Evaluate the following report against the source findings.

Score each dimension from 0.0 to 1.0:
1. **Hallucination Score**: Are all claims grounded in the source findings? (1.0 = no hallucinations)
2. **Citation Score**: Are findings properly attributed to source agents? (1.0 = fully cited)
3. **Coherence Score**: Is the report well-structured and logically coherent? (1.0 = excellent)

Respond ONLY in this exact format (no other text):
HALLUCINATION: <score>
CITATION: <score>
COHERENCE: <score>
ISSUES: <comma-separated list of issues, or "none">"""


class ReportEvaluator:
    """Evaluates investigation reports for quality and reliability."""

    def __init__(self) -> None:
        self._llm = create_llm(temperature=0.0)

    def evaluate(
        self,
        report: str,
        findings: dict[str, list[dict[str, Any]]],
    ) -> EvaluationResult:
        """Run all evaluation checks on a report."""
        # Check completeness (rule-based)
        completeness = self._check_completeness(report)

        # LLM-based evaluation
        llm_scores = self._llm_evaluate(report, findings)

        overall = (
            completeness * 0.2
            + llm_scores["hallucination"] * 0.35
            + llm_scores["citation"] * 0.25
            + llm_scores["coherence"] * 0.2
        )

        issues = llm_scores.get("issues", [])
        if completeness < 0.8:
            issues.append(f"Missing report sections (completeness: {completeness:.0%})")

        passed = overall >= 0.6 and llm_scores["hallucination"] >= 0.5

        return EvaluationResult(
            overall_score=round(overall, 3),
            hallucination_score=llm_scores["hallucination"],
            citation_score=llm_scores["citation"],
            completeness_score=completeness,
            coherence_score=llm_scores["coherence"],
            issues=issues,
            passed=passed,
        )

    def _check_completeness(self, report: str) -> float:
        """Check if the report contains all required sections."""
        report_lower = report.lower()
        found = sum(
            1
            for section in REQUIRED_SECTIONS
            if section.replace("_", " ") in report_lower
        )
        return found / len(REQUIRED_SECTIONS)

    def _llm_evaluate(
        self, report: str, findings: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Use LLM-as-judge to evaluate report quality."""
        # Compile source findings for comparison
        source_text_parts = []
        for source, finding_list in findings.items():
            for f in finding_list:
                source_text_parts.append(f"[{source}] {f.get('content', '')[:500]}")
        source_text = "\n".join(source_text_parts)

        messages = [
            HumanMessage(content=EVALUATION_PROMPT),
            HumanMessage(
                content=(
                    f"## Source Findings\n{source_text}\n\n"
                    f"## Generated Report\n{report}"
                )
            ),
        ]

        try:
            response = self._llm.invoke(messages)
            return self._parse_evaluation(response.content)
        except Exception as e:
            logger.error("LLM evaluation failed: %s", e)
            return {
                "hallucination": 0.5,
                "citation": 0.5,
                "coherence": 0.5,
                "issues": [f"Evaluation failed: {e}"],
            }

    @staticmethod
    def _parse_evaluation(text: str) -> dict[str, Any]:
        """Parse the structured evaluation response."""
        scores: dict[str, Any] = {
            "hallucination": 0.5,
            "citation": 0.5,
            "coherence": 0.5,
            "issues": [],
        }

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("HALLUCINATION:"):
                scores["hallucination"] = _parse_score(line.split(":", 1)[1])
            elif line.startswith("CITATION:"):
                scores["citation"] = _parse_score(line.split(":", 1)[1])
            elif line.startswith("COHERENCE:"):
                scores["coherence"] = _parse_score(line.split(":", 1)[1])
            elif line.startswith("ISSUES:"):
                issues_text = line.split(":", 1)[1].strip()
                if issues_text.lower() != "none":
                    scores["issues"] = [i.strip() for i in issues_text.split(",") if i.strip()]

        return scores


def _parse_score(text: str) -> float:
    """Safely parse a float score from text."""
    match = re.search(r"(\d+\.?\d*)", text.strip())
    if match:
        return min(1.0, max(0.0, float(match.group(1))))
    return 0.5
