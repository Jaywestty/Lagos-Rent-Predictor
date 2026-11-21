
from loguru import logger
from tavily import TavilyClient

from config import TAVILY_API_KEY
from observability.events import tavily_search
from observability.tracer import record_call

MAX_SEARCH_RESULTS = 5


class SearchError(Exception):
    pass


def _get_client() -> TavilyClient:
    if not TAVILY_API_KEY:
        raise SearchError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=TAVILY_API_KEY)


def search_advice(query: str, purpose: str = "advice", max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    if not query or not query.strip():
        raise SearchError("query text is empty")

    client = _get_client()
    record_call("tavily")
    try:
        response = tavily_search(
            client,
            purpose,
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
        )
    except Exception as exc:
        logger.error("Tavily search failed for query='{}': {}", query, exc)
        raise SearchError(f"web search failed: {exc}") from exc

    results = response.get("results", [])
    logger.info("Tavily returned {} result(s) for query='{}'", len(results), query)

    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "content": r.get("content"),
        }
        for r in results
        if r.get("content")
    ]
