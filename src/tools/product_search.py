"""Product discovery tools — used by shopping agents.

Mirrors the pattern of azure_search.py / web_search.py: real implementations
hit external services (e-commerce APIs, scrapers, price aggregators), but in
demo mode (`MOCK_TOOLS=true`) return curated fixtures so the agent
orchestration and Langfuse tracing can be showcased without integration setup.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from src.tools._mock_data import (
    SHOPPING_PRICE_HISTORY,
    SHOPPING_PRODUCTS,
    SHOPPING_REVIEWS,
    mock_enabled,
)

logger = logging.getLogger(__name__)


@tool
def search_products(query: str, max_price_chf: float = 200.0) -> str:
    """Search for products matching a query, with an optional price ceiling.

    Returns product name, current price, key specs, and source URL for each match.

    Args:
        query: Description of the product category or features to search for.
        max_price_chf: Maximum price in CHF. Defaults to 200.
    """
    logger.info("[%s] search_products query=%r max=%s", "MOCK" if mock_enabled() else "LIVE", query, max_price_chf)

    if not mock_enabled():
        return (
            "Live product search is not wired in this build. "
            "Set MOCK_TOOLS=true or implement an integration with a product catalog API."
        )

    matches = [p for p in SHOPPING_PRODUCTS if p["price_chf"] <= max_price_chf * 1.5]
    if not matches:
        return f"No products found matching '{query}' under CHF {max_price_chf}."

    formatted = []
    for p in matches:
        specs = ", ".join(f"{k}={v}" for k, v in p["specs"].items())
        formatted.append(
            f"**{p['product_name']}** — CHF {p['price_chf']:.2f}\n"
            f"Category: {p['category']}\n"
            f"Specs: {specs}\n"
            f"URL: {p['source_url']}"
        )
    return "\n\n---\n\n".join(formatted)


@tool
def fetch_reviews(product_name: str) -> str:
    """Fetch and aggregate professional reviews for a specific product.

    Returns reviewer source, rating, and a short summary for each review found.
    If no exact match is found, returns reviews for the closest product names.

    Args:
        product_name: The product to find reviews for.
    """
    logger.info("[%s] fetch_reviews product=%r", "MOCK" if mock_enabled() else "LIVE", product_name)

    if not mock_enabled():
        return (
            "Live review aggregation is not wired in this build. "
            "Set MOCK_TOOLS=true or implement scrapers / review-API integration."
        )

    needle = product_name.lower()
    matches = [r for r in SHOPPING_REVIEWS if needle in r["product_name"].lower()]

    if not matches:
        # loosen the match: try the first word if no exact substring hit
        first_word = needle.split()[0] if needle.split() else ""
        matches = [r for r in SHOPPING_REVIEWS if first_word and first_word in r["product_name"].lower()]

    if not matches:
        return f"No reviews found for '{product_name}'."

    formatted = []
    for r in matches:
        formatted.append(
            f"**{r['source']} review of {r['product_name']}** — {r['rating']}/5\n"
            f"{r['summary']}\n"
            f"URL: {r['url']}"
        )
    return "\n\n---\n\n".join(formatted)


@tool
def compare_prices(product_name: str) -> str:
    """Compare current prices and 30-day price history for a product across retailers.

    Returns current price, 30-day low, 30-day average, trend direction, and
    best retailer.

    Args:
        product_name: The product to look up pricing for.
    """
    logger.info("[%s] compare_prices product=%r", "MOCK" if mock_enabled() else "LIVE", product_name)

    if not mock_enabled():
        return (
            "Live price comparison is not wired in this build. "
            "Set MOCK_TOOLS=true or implement a price-aggregator integration."
        )

    needle = product_name.lower()
    matches = [p for p in SHOPPING_PRICE_HISTORY if needle in p["product_name"].lower()]

    if not matches:
        return f"No price history found for '{product_name}'."

    formatted = []
    for p in matches:
        savings = p["average_30d_chf"] - p["current_chf"]
        savings_str = f" (CHF {savings:.2f} below 30d avg)" if savings > 0 else ""
        formatted.append(
            f"**{p['product_name']}**\n"
            f"  Current: CHF {p['current_chf']:.2f}{savings_str}\n"
            f"  30-day low: CHF {p['lowest_30d_chf']:.2f}\n"
            f"  30-day avg: CHF {p['average_30d_chf']:.2f}\n"
            f"  Trend: {p['trend']}\n"
            f"  Best retailer: {p['best_retailer']}"
        )
    return "\n\n---\n\n".join(formatted)
