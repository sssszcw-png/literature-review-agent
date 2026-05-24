"""arXiv OAI-PMH API client with XML response parsing."""

import asyncio
import logging
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

import aiohttp

from src.config.settings import Settings
from src.search.models import SearchResult
from src.utils.async_utils import RateLimiter

logger = logging.getLogger(__name__)

ARXIV_NAMESPACE = "http://www.w3.org/2005/Atom"


class ArxivClient:
    """Client for the arXiv API (free, no key required)."""

    def __init__(self, settings: Settings):
        self.base_url = settings.arxiv_base_url.rstrip("/")
        self.timeout = settings.search_timeout
        self.max_results = settings.search_max_results_per_query
        self.rate_limiter = RateLimiter(1.0)  # arXiv asks for ~1 req/s

    async def search(
        self, query: str, max_results: int | None = None
    ) -> list[SearchResult]:
        """Search arXiv for papers matching the query."""
        limit = max_results or self.max_results

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(limit, 100),
            "sortBy": "relevance",
        }
        url = f"{self.base_url}?{urlencode(params)}"

        await self.rate_limiter.acquire()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    text = await resp.text()
        except asyncio.TimeoutError:
            logger.warning(f"arXiv search timed out for: {query}")
            return []
        except aiohttp.ClientError as e:
            logger.warning(f"arXiv search request failed: {e}")
            return []

        return self._parse_atom_response(text, query)

    def _parse_atom_response(self, xml_text: str, query: str) -> list[SearchResult]:
        """Parse arXiv ATOM XML response into SearchResults."""
        results = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"arXiv XML parse error: {e}")
            return []

        ns = {"atom": ARXIV_NAMESPACE}
        entries = root.findall("atom:entry", ns)

        for entry in entries:
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

            arxiv_url = ""
            pdf_url = ""
            arxiv_id = ""
            for link in entry.findall("atom:link", ns):
                rel = link.get("rel", "")
                href = link.get("href", "")
                if rel == "alternate":
                    arxiv_url = href
                elif "pdf" in link.get("title", "").lower():
                    pdf_url = href

            # Extract arXiv ID from URL
            if arxiv_url:
                arxiv_id = arxiv_url.split("/abs/")[-1].rstrip("/")
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            authors = []
            for author_elem in entry.findall("atom:author", ns):
                name_elem = author_elem.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            summary_elem = entry.find("atom:summary", ns)
            abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""

            published = entry.find("atom:published", ns)
            year = int(published.text[:4]) if published is not None and published.text else None

            results.append(
                SearchResult(
                    paper_id=f"arxiv:{arxiv_id}" if arxiv_id else "",
                    title=title,
                    authors=authors,
                    year=year,
                    url=arxiv_url,
                    venue="arXiv preprint",
                    citation_count=0,
                    abstract=abstract,
                    source="arxiv",
                    pdf_url=pdf_url,
                    doi="",
                )
            )

        logger.info(f"arXiv: '{query}' → {len(results)} results")
        return results
