"""Thin wrapper around Tavily so the rest of the codebase never touches the SDK directly."""
from __future__ import annotations
from typing import List
import logging

from tavily import TavilyClient
from graph.config import TAVILY_API_KEY
from graph.state import EvidenceChunk

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://tavily.com"
            )
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


def web_search(query: str, sub_question_id: str, max_results: int = 4) -> List[EvidenceChunk]:
    """
    Search the web via Tavily and return normalized EvidenceChunks.
    Returns an empty list (never raises) on API failure or empty results --
    callers should treat "no web evidence" as a valid, expected state, since
    the vector store may still cover the sub-question.
    """
    try:
        client = _get_client()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any network/API failure
        logger.warning("Tavily search failed for %r: %s", query, exc)
        return []

    results = response.get("results", []) if isinstance(response, dict) else []
    if not results:
        logger.info("Tavily returned zero results for %r", query)

    chunks: List[EvidenceChunk] = []
    for r in results:
        chunks.append(
            EvidenceChunk(
                sub_question_id=sub_question_id,
                source_type="web",
                source_id=r.get("url", "unknown-url"),
                title=r.get("title", "Untitled"),
                text=(r.get("content") or "")[:2000],  # cap length fed into the LLM
                relevance_score=r.get("score"),
            )
        )
    return chunks
