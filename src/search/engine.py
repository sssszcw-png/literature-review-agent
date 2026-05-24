"""Search engine orchestrator: concurrent search, deduplication, and merge."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.config.settings import Settings
from src.search.arxiv import ArxivClient
from src.search.models import SearchResult
from src.search.semantic_scholar import SemanticScholarClient
from src.utils.async_utils import bounded_gather
from src.utils.dedup import make_dedup_key, titles_match

logger = logging.getLogger(__name__)


class SearchEngine:
    """Orchestrates concurrent search across Semantic Scholar and arXiv, with dedup."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ss_client = SemanticScholarClient(settings)
        self.arxiv_client = ArxivClient(settings)
        self.max_concurrent = settings.max_concurrent_searches

    async def search(
        self,
        queries: list[str],
        max_per_query: int | None = None,
    ) -> list[SearchResult]:
        """Execute multiple queries concurrently across both APIs, deduplicate results."""

        # Launch all searches concurrently
        coros = []
        for q in queries:
            coros.append(self.ss_client.search(q, max_per_query))
            coros.append(self.arxiv_client.search(q, max_per_query))

        results_lists = await bounded_gather(
            *coros,
            max_concurrent=self.max_concurrent,
            return_exceptions=True,
        )

        all_results: list[SearchResult] = []
        for result in results_lists:
            if isinstance(result, Exception):
                logger.warning(f"Search failed: {result}")
                continue
            all_results.extend(result)

        return self._deduplicate(all_results)

    async def search_single(
        self,
        query: str,
        max_per_query: int | None = None,
    ) -> list[SearchResult]:
        """Execute a single query across both APIs."""
        results = await self.search([query], max_per_query)
        return results

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate papers using normalized title + first-author surname matching."""
        seen: dict[str, SearchResult] = {}

        for r in results:
            if not r.title:
                continue

            # Try exact dedup key first
            key = make_dedup_key(r.title, r.first_author_surname(), str(r.year or ""))
            if key in seen:
                existing = seen[key]
                existing = self._merge_result(existing, r)
                seen[key] = existing
                continue

            # Fuzzy check against existing titles
            found = False
            for existing_key, existing_r in seen.items():
                if titles_match(r.title, existing_r.title):
                    seen[existing_key] = self._merge_result(existing_r, r)
                    found = True
                    break

            if not found:
                seen[key] = r

        merged = sorted(
            seen.values(),
            key=lambda x: (x.citation_count, x.year or 0),
            reverse=True,
        )
        logger.info(f"Dedup: {len(results)} raw → {len(merged)} unique papers")
        return merged

    @staticmethod
    def _merge_result(a: SearchResult, b: SearchResult) -> SearchResult:
        """Merge two results for the same paper, preferring non-empty fields."""
        a.abstract = a.abstract or b.abstract
        a.pdf_url = a.pdf_url or b.pdf_url
        a.doi = a.doi or b.doi
        a.citation_count = max(a.citation_count, b.citation_count)
        a.url = a.url or b.url
        if not a.authors and b.authors:
            a.authors = b.authors
        if not a.year and b.year:
            a.year = b.year
        if not a.venue or a.venue == "arXiv preprint":
            a.venue = b.venue
        return a
