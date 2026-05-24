"""Semantic Scholar Academic Graph API client."""

import asyncio
import logging
from typing import Optional

import aiohttp

from src.config.settings import Settings
from src.search.models import SearchResult
from src.utils.async_utils import RateLimiter

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """Client for the Semantic Scholar Academic Graph API (free, no key required)."""

    BASE_PATH = "/graph/v1"

    def __init__(self, settings: Settings):
        self.base_url = settings.semantic_scholar_base_url.rstrip("/")
        self.timeout = settings.search_timeout
        self.max_results = settings.search_max_results_per_query
        self.rate_limiter = RateLimiter(settings.semantic_scholar_rate_limit)

    async def search(
        self, query: str, max_results: int | None = None
    ) -> list[SearchResult]:
        """Search papers by keyword query. Returns structured SearchResults."""
        limit = max_results or self.max_results

        await self.rate_limiter.acquire()

        fields = "paperId,externalIds,title,authors,year,venue,citationCount,abstract,openAccessPdf,publicationTypes"
        url = f"{self.base_url}/paper/search"
        params: dict = {
            "query": query,
            "limit": min(limit, 100),
            "fields": fields,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    data = await resp.json()
        except asyncio.TimeoutError:
            logger.warning(f"Semantic Scholar search timed out for: {query}")
            return []
        except aiohttp.ClientError as e:
            logger.warning(f"Semantic Scholar search request failed: {e}")
            return []

        if "data" not in data:
            logger.warning(f"Unexpected Semantic Scholar response: {data}")
            return []

        results = []
        for item in data["data"]:
            paper_id = item.get("paperId", "")
            if not paper_id:
                continue

            external_ids = item.get("externalIds") or {}
            doi = external_ids.get("DOI", "")
            arxiv_id = external_ids.get("ArXiv", "")

            authors = [a.get("name", "") for a in item.get("authors", [])]

            pdf_url = ""
            open_access = item.get("openAccessPdf")
            if open_access and open_access.get("url"):
                pdf_url = open_access["url"]

            results.append(
                SearchResult(
                    paper_id=f"ss:{paper_id}",
                    title=item.get("title", ""),
                    authors=authors,
                    year=item.get("year"),
                    venue=item.get("venue", ""),
                    citation_count=item.get("citationCount", 0) or 0,
                    abstract=item.get("abstract", "") or "",
                    source="semantic_scholar",
                    pdf_url=pdf_url,
                    doi=doi,
                    url=f"https://api.semanticscholar.org/paper/{paper_id}",
                )
            )

        logger.info(
            f"Semantic Scholar: '{query}' → {len(results)} results"
        )
        return results

    async def get_paper_by_id(self, paper_id: str) -> SearchResult | None:
        """Fetch a single paper by its Semantic Scholar ID."""
        ss_id = paper_id.removeprefix("ss:")
        fields = "paperId,externalIds,title,authors,year,venue,citationCount,abstract,openAccessPdf"
        url = f"{self.base_url}/paper/{ss_id}"
        params = {"fields": fields}

        await self.rate_limiter.acquire()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    data = await resp.json()
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.warning(f"Semantic Scholar get_paper failed: {e}")
            return None

        if "paperId" not in data:
            return None

        external_ids = data.get("externalIds") or {}
        pdf_url = ""
        oa = data.get("openAccessPdf")
        if oa and oa.get("url"):
            pdf_url = oa["url"]

        authors = [a.get("name", "") for a in data.get("authors", [])]

        return SearchResult(
            paper_id=f"ss:{data['paperId']}",
            title=data.get("title", ""),
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue", ""),
            citation_count=data.get("citationCount", 0) or 0,
            abstract=data.get("abstract", "") or "",
            source="semantic_scholar",
            pdf_url=pdf_url,
            doi=external_ids.get("DOI", ""),
            url=f"https://api.semanticscholar.org/paper/{data['paperId']}",
        )
