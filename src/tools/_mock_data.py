"""Fixture data for tool mocking in demo runs.

Activated by setting `MOCK_TOOLS=true` in the environment. Lets demos run without
provisioning Azure AI Search, Bing, Tavily, or any external retrieval service —
useful for portfolio runs where the goal is to showcase agent orchestration and
observability rather than data integration.
"""

from __future__ import annotations

import os


def mock_enabled() -> bool:
    return os.getenv("MOCK_TOOLS", "").lower() in ("1", "true", "yes")


# ─────────────────────────────────────────────────────────────────────────────
# investigation domain — tesla q3 2024 financial risk investigation
# ─────────────────────────────────────────────────────────────────────────────

INVESTIGATION_DOCS = [
    {
        "document_name": "Tesla_10Q_Q3_2024.pdf",
        "chunk_index": 12,
        "score": 0.91,
        "content": (
            "Automotive gross margin declined to 17.05% in Q3 2024 from 19.6% a year prior, "
            "reflecting continued ASP pressure from Model 3/Y price reductions in North America "
            "and Europe. Management attributes ~180bps of the decline to fixed-cost absorption "
            "as Cybertruck ramped below plan."
        ),
    },
    {
        "document_name": "Tesla_10Q_Q3_2024.pdf",
        "chunk_index": 27,
        "score": 0.84,
        "content": (
            "Free cash flow of $2.74B for the quarter was supported by a $1.4B working-capital "
            "release, primarily from inventory drawdown. Excluding this benefit, normalized FCF "
            "would have been approximately $1.3B — below consensus of $1.6B."
        ),
    },
    {
        "document_name": "Tesla_Earnings_Call_Transcript_Q3_2024.pdf",
        "chunk_index": 4,
        "score": 0.78,
        "content": (
            "CFO commentary acknowledged regulatory credit revenue of $739M was 'meaningfully "
            "above the run-rate' and may not repeat in Q4. Excluding credits, automotive gross "
            "margin would have been ~14.9%, the lowest since 2017."
        ),
    },
]

INVESTIGATION_NEWS = [
    {
        "name": "Tesla shares jump after Q3 beat, but margin questions linger",
        "url": "https://example.com/news/tesla-q3-2024-margin",
        "snippet": (
            "Tesla reported Q3 EPS of $0.72, beating consensus, but analysts focused on the "
            "17% automotive gross margin and heavy reliance on regulatory credits."
        ),
        "provider": "Reuters",
        "date": "2024-10-23",
    },
    {
        "name": "NHTSA opens probe into Tesla Full Self-Driving after fatal crashes",
        "url": "https://example.com/news/nhtsa-fsd-probe",
        "snippet": (
            "The agency is investigating 2.4M Tesla vehicles equipped with FSD following four "
            "reported collisions in low-visibility conditions, including one fatality."
        ),
        "provider": "NYT",
        "date": "2024-10-18",
    },
    {
        "name": "Tesla cuts Cybertruck production targets for 2025",
        "url": "https://example.com/news/cybertruck-cut",
        "snippet": (
            "Internal memo seen by Bloomberg shows Cybertruck output reduced by ~30% versus "
            "prior plan, citing demand softness and persistent assembly defects."
        ),
        "provider": "Bloomberg",
        "date": "2024-10-15",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# formatters — produce strings shaped like the real tools' outputs
# ─────────────────────────────────────────────────────────────────────────────


def format_investigation_docs() -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d['document_name']}, Chunk #{d['chunk_index']}] "
        f"(Score: {d['score']:.3f})\n{d['content']}"
        for d in INVESTIGATION_DOCS
    )


def format_investigation_news() -> str:
    return "\n\n---\n\n".join(
        f"**{n['name']}**\n"
        f"Source: {n['provider']} | Published: {n['date']}\n"
        f"URL: {n['url']}\n"
        f"{n['snippet']}"
        for n in INVESTIGATION_NEWS
    )


def format_investigation_web() -> str:
    return "\n\n---\n\n".join(
        f"**{n['name']}**\nURL: {n['url']}\n{n['snippet']}"
        for n in INVESTIGATION_NEWS
    )
