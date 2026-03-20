"""Web search tool using Bing Search API."""

from __future__ import annotations

import logging

import httpx
from langchain_core.tools import tool

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@tool
def web_search(query: str, count: int = 5) -> str:
    """Search the web for relevant information using Bing Search API.

    Use this tool to find current information about entities, events,
    regulations, or any topic relevant to the investigation.

    Args:
        query: The search query.
        count: Number of results to return (max 10).
    """
    settings = get_settings()
    if not settings.bing_search_api_key:
        return "Web search is not configured (missing Bing API key)."

    count = min(count, 10)

    headers = {"Ocp-Apim-Subscription-Key": settings.bing_search_api_key}
    params = {"q": query, "count": str(count), "textFormat": "Raw"}

    try:
        response = httpx.get(
            settings.bing_search_endpoint,
            headers=headers,
            params=params,
            timeout=15.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("Web search failed: %s", e)
        return f"Web search failed: {e}"

    data = response.json()
    web_pages = data.get("webPages", {}).get("value", [])

    if not web_pages:
        return "No web results found for this query."

    formatted = []
    for page in web_pages:
        formatted.append(
            f"**{page['name']}**\n"
            f"URL: {page['url']}\n"
            f"{page.get('snippet', 'No snippet available.')}"
        )

    return "\n\n---\n\n".join(formatted)


@tool
def web_search_news(query: str, count: int = 5) -> str:
    """Search for recent news articles related to the investigation.

    Use this tool to find recent news about companies, events,
    or topics under investigation.

    Args:
        query: The news search query.
        count: Number of results to return (max 10).
    """
    settings = get_settings()
    if not settings.bing_search_api_key:
        return "News search is not configured (missing Bing API key)."

    count = min(count, 10)

    headers = {"Ocp-Apim-Subscription-Key": settings.bing_search_api_key}
    params = {"q": query, "count": str(count), "textFormat": "Raw"}

    # Use the news endpoint
    news_endpoint = settings.bing_search_endpoint.replace("/search", "/news/search")

    try:
        response = httpx.get(
            news_endpoint,
            headers=headers,
            params=params,
            timeout=15.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("News search failed: %s", e)
        return f"News search failed: {e}"

    data = response.json()
    articles = data.get("value", [])

    if not articles:
        return "No news articles found for this query."

    formatted = []
    for article in articles:
        published = article.get("datePublished", "Unknown date")
        provider = article.get("provider", [{}])[0].get("name", "Unknown")
        formatted.append(
            f"**{article['name']}**\n"
            f"Source: {provider} | Published: {published}\n"
            f"URL: {article.get('url', 'N/A')}\n"
            f"{article.get('description', 'No description available.')}"
        )

    return "\n\n---\n\n".join(formatted)
