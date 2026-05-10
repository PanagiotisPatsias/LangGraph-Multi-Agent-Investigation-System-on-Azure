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
# shopping domain — wireless earbuds under chf 200 product discovery
# ─────────────────────────────────────────────────────────────────────────────

SHOPPING_PRODUCTS = [
    {
        "product_name": "Sony WF-1000XM5",
        "category": "wireless earbuds",
        "price_chf": 269.00,
        "specs": {
            "battery_hours": 8,
            "anc": "Industry-leading active noise cancellation",
            "codecs": ["LDAC", "AAC", "SBC"],
            "ip_rating": "IPX4",
            "weight_g": 5.9,
        },
        "source_url": "https://example.com/products/sony-wf-1000xm5",
    },
    {
        "product_name": "Apple AirPods Pro 2 (USB-C)",
        "category": "wireless earbuds",
        "price_chf": 249.00,
        "specs": {
            "battery_hours": 6,
            "anc": "Adaptive ANC, Transparency mode",
            "codecs": ["AAC"],
            "ip_rating": "IP54",
            "weight_g": 5.3,
        },
        "source_url": "https://example.com/products/airpods-pro-2",
    },
    {
        "product_name": "Bose QuietComfort Ultra Earbuds",
        "category": "wireless earbuds",
        "price_chf": 299.00,
        "specs": {
            "battery_hours": 6,
            "anc": "Best-in-class ANC with CustomTune calibration",
            "codecs": ["aptX Adaptive", "AAC", "SBC"],
            "ip_rating": "IPX4",
            "weight_g": 6.2,
        },
        "source_url": "https://example.com/products/bose-qc-ultra",
    },
    {
        "product_name": "Nothing Ear (2)",
        "category": "wireless earbuds",
        "price_chf": 149.00,
        "specs": {
            "battery_hours": 6.3,
            "anc": "Active Noise Cancellation, dual-chamber design",
            "codecs": ["LHDC 5.0", "AAC", "SBC"],
            "ip_rating": "IP54",
            "weight_g": 4.5,
        },
        "source_url": "https://example.com/products/nothing-ear-2",
    },
    {
        "product_name": "Anker Soundcore Liberty 4 NC",
        "category": "wireless earbuds",
        "price_chf": 99.00,
        "specs": {
            "battery_hours": 10,
            "anc": "Adaptive ANC 2.0 — strong for the price",
            "codecs": ["LDAC", "AAC", "SBC"],
            "ip_rating": "IPX4",
            "weight_g": 5.0,
        },
        "source_url": "https://example.com/products/anker-liberty-4-nc",
    },
]

SHOPPING_REVIEWS = [
    {
        "product_name": "Nothing Ear (2)",
        "source": "WhatHiFi",
        "rating": 4.5,
        "summary": (
            "Excellent sound quality with LHDC support, comfortable fit, and competitive ANC "
            "for the price. Companion app is intuitive."
        ),
        "url": "https://example.com/reviews/nothing-ear-2-whathifi",
    },
    {
        "product_name": "Nothing Ear (2)",
        "source": "RTINGS.com",
        "rating": 7.8,
        "summary": (
            "Above-average noise cancellation and good build, though battery life with ANC "
            "dips below 5h. Best value in the sub-CHF 200 segment."
        ),
        "url": "https://example.com/reviews/nothing-ear-2-rtings",
    },
    {
        "product_name": "Anker Soundcore Liberty 4 NC",
        "source": "TechRadar",
        "rating": 4.0,
        "summary": (
            "Remarkable feature set for under CHF 100 — LDAC, 10h battery, solid ANC. Sound "
            "is bass-heavy by default but EQ is tunable."
        ),
        "url": "https://example.com/reviews/anker-liberty-4-nc-techradar",
    },
    {
        "product_name": "Apple AirPods Pro 2 (USB-C)",
        "source": "The Verge",
        "rating": 4.5,
        "summary": (
            "Best-in-class for iPhone users thanks to seamless integration, but locked-in "
            "ecosystem and no LDAC/aptX limit appeal for Android."
        ),
        "url": "https://example.com/reviews/airpods-pro-2-verge",
    },
    {
        "product_name": "Sony WF-1000XM5",
        "source": "WhatHiFi",
        "rating": 5.0,
        "summary": (
            "Outstanding sound, leading ANC, and LDAC support. Premium price but the "
            "audiophile choice in true wireless."
        ),
        "url": "https://example.com/reviews/sony-wf-1000xm5-whathifi",
    },
]

SHOPPING_PRICE_HISTORY = [
    {
        "product_name": "Nothing Ear (2)",
        "current_chf": 149.00,
        "lowest_30d_chf": 129.00,
        "average_30d_chf": 142.50,
        "trend": "stable",
        "best_retailer": "Digitec",
    },
    {
        "product_name": "Anker Soundcore Liberty 4 NC",
        "current_chf": 99.00,
        "lowest_30d_chf": 79.00,
        "average_30d_chf": 89.00,
        "trend": "rising — Black Friday discount expired",
        "best_retailer": "Galaxus",
    },
    {
        "product_name": "Apple AirPods Pro 2 (USB-C)",
        "current_chf": 249.00,
        "lowest_30d_chf": 229.00,
        "average_30d_chf": 245.00,
        "trend": "stable",
        "best_retailer": "Interdiscount",
    },
    {
        "product_name": "Sony WF-1000XM5",
        "current_chf": 269.00,
        "lowest_30d_chf": 249.00,
        "average_30d_chf": 262.00,
        "trend": "declining",
        "best_retailer": "Microspot",
    },
    {
        "product_name": "Bose QuietComfort Ultra Earbuds",
        "current_chf": 299.00,
        "lowest_30d_chf": 279.00,
        "average_30d_chf": 295.00,
        "trend": "stable",
        "best_retailer": "Digitec",
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
