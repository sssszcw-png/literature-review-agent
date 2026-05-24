"""Node: search — execute search queries across academic APIs."""

from __future__ import annotations

import logging

from src.agent.state import AgentState
from src.config.settings import get_settings
from src.search.engine import SearchEngine
from src.search.models import SearchResult

logger = logging.getLogger(__name__)


async def search_node(state: AgentState) -> AgentState:
    """Execute search queries concurrently and populate paper_index.

    Reads from state: search_queries, paper_index, max_papers_per_round
    Writes to state: paper_index, search_results
    """
    queries = state.get("search_queries", [])
    if not queries:
        logger.warning("No search queries to execute")
        return state

    settings = get_settings()
    engine = SearchEngine(settings)

    results = await engine.search(queries, max_per_query=settings.search_max_results_per_query)

    # Populate paper_index — only add truly new papers
    paper_index = state.get("paper_index", {})

    max_papers = settings.max_papers_per_round
    new_papers = results[:max_papers]

    new_ids = []
    for r in new_papers:
        if r.paper_id not in paper_index:
            paper_dict = _search_result_to_dict(r)
            paper_index[r.paper_id] = paper_dict
            new_ids.append(r.paper_id)
        # Skip already-indexed papers to avoid re-processing

    state["paper_index"] = paper_index
    state["search_results"] = new_ids

    errors = state.get("errors", [])
    if len(results) > max_papers:
        errors.append({
            "node": "search",
            "message": f"Capped results from {len(results)} to {max_papers}",
            "level": "info",
        })
        state["errors"] = errors

    logger.info(
        f"[search] {len(queries)} queries → {len(results)} raw results → "
        f"{len(new_ids)} new papers added to index (total: {len(paper_index)})"
    )
    return state


def _search_result_to_dict(r: SearchResult) -> dict:
    return {
        "paper_id": r.paper_id,
        "title": r.title,
        "authors": r.authors,
        "year": r.year,
        "url": r.url,
        "venue": r.venue,
        "citation_count": r.citation_count,
        "abstract": r.abstract,
        "source": r.source,
        "pdf_url": r.pdf_url,
        "doi": r.doi,
        "full_text_available": bool(r.pdf_url),
    }
