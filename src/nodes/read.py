"""Node: read — read paper content (abstracts in Round 1, full-text in Round 2+)."""

from __future__ import annotations

import asyncio
import logging

from src.agent.state import AgentState
from src.config.constants import PHASE_BROAD
from src.config.settings import get_settings
from src.pdf.parser import PaperParser

logger = logging.getLogger(__name__)


async def read_node(state: AgentState) -> AgentState:
    """Read papers: abstracts only in Round 1, full-text PDFs in Round 2+.

    Reads from state: search_results, paper_index, phase, papers_read
    Writes to state: papers_read, read_failures, paper_index (updates full_text_available)
    """
    search_results = state.get("search_results", [])
    phase = state.get("phase", PHASE_BROAD)
    paper_index = state.get("paper_index", {})
    papers_read = state.get("papers_read", {})
    read_failures = state.get("read_failures", {})

    if not search_results:
        logger.warning("No search results to read")
        return state

    settings = get_settings()
    parser = PaperParser(settings)

    if phase == PHASE_BROAD:
        # Round 1: abstracts only (fast)
        for paper_id in search_results:
            if paper_id in papers_read:
                continue  # Skip already read
            paper = paper_index.get(paper_id, {})
            abstract = paper.get("abstract", "")
            papers_read[paper_id] = abstract
            paper["full_text_available"] = False
            paper_index[paper_id] = paper
    else:
        # Round 2+: full text via PDF parsing, concurrent across papers
        async def _parse_one(paper_id: str) -> None:
            if paper_id in papers_read:
                return
            paper = paper_index.get(paper_id, {})
            pdf_url = paper.get("pdf_url", "")
            abstract = paper.get("abstract", "")

            if not pdf_url:
                papers_read[paper_id] = abstract
                return

            result = await parser.parse(
                paper_id=paper_id,
                pdf_url=pdf_url,
                abstract=abstract,
            )
            if result.success:
                papers_read[paper_id] = result.text
                paper["full_text_available"] = result.full_text_available
                paper_index[paper_id] = paper
            else:
                read_failures[paper_id] = result.error
                if abstract:
                    papers_read[paper_id] = abstract

        # Filter unread papers with PDF URLs
        unread = [
            pid for pid in search_results
            if pid not in papers_read
        ]
        if unread:
            tasks = [asyncio.create_task(_parse_one(pid)) for pid in unread]
            await asyncio.gather(*tasks, return_exceptions=True)

    state["papers_read"] = papers_read
    state["read_failures"] = read_failures
    state["paper_index"] = paper_index

    logger.info(
        f"[read] phase={phase}: {len(search_results)} papers, "
        f"{len(read_failures)} failures"
    )
    return state
