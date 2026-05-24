"""PDF parser orchestrator with degradation fallback chain."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import Settings
from src.pdf.cache import PaperCache
from src.pdf.downloader import PDFDownloader
from src.pdf.docling_parser import DoclingParser
from src.pdf.marker_parser import MarkerParser

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing a PDF paper."""

    paper_id: str
    success: bool
    text: str = ""
    method: str = ""  # "docling", "marker", "abstract_only", "cache"
    full_text_available: bool = False
    error: str = ""


class PaperParser:
    """Orchestrates PDF download and parsing with fallback chain.

    Chain: Cache → Docling → Marker → Abstract-only
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache = PaperCache(settings.pdf_cache_dir)
        self.downloader = PDFDownloader(settings)
        self.docling = DoclingParser()
        self.marker = MarkerParser()

    async def parse(
        self,
        paper_id: str,
        pdf_url: str = "",
        abstract: str = "",
    ) -> ParseResult:
        """Parse a paper's full text with full degradation chain.

        1. Return cached version if available
        2. Download PDF and parse with Docling
        3. Fall back to Marker
        4. Fall back to abstract only
        """

        # 1. Check cache
        cached = self.cache.get(paper_id)
        if cached:
            return ParseResult(
                paper_id=paper_id,
                success=True,
                text=cached,
                method="cache",
                full_text_available=True,
            )

        # 2. & 3. Download + parse
        if pdf_url:
            pdf_path = await self.downloader.download(pdf_url, paper_id)
            if pdf_path:
                # Try Docling
                text = await self.docling.parse(pdf_path)
                if text:
                    self.cache.put(paper_id, text)
                    self.downloader.cleanup(paper_id)
                    return ParseResult(
                        paper_id=paper_id,
                        success=True,
                        text=text,
                        method="docling",
                        full_text_available=True,
                    )

                # Try Marker
                text = await self.marker.parse(pdf_path)
                if text:
                    self.cache.put(paper_id, text)
                    self.downloader.cleanup(paper_id)
                    return ParseResult(
                        paper_id=paper_id,
                        success=True,
                        text=text,
                        method="marker",
                        full_text_available=True,
                    )

                self.downloader.cleanup(paper_id)

        # 4. Abstract only
        if abstract:
            logger.info(f"Falling back to abstract-only for {paper_id}")
            return ParseResult(
                paper_id=paper_id,
                success=True,
                text=abstract,
                method="abstract_only",
                full_text_available=False,
            )

        return ParseResult(
            paper_id=paper_id,
            success=False,
            method="",
            full_text_available=False,
            error="No PDF URL available and no abstract provided",
        )

    async def parse_abstract_only(self, paper_id: str, abstract: str) -> ParseResult:
        """Parse paper using only its abstract (Round 1 mode)."""
        return ParseResult(
            paper_id=paper_id,
            success=True,
            text=abstract,
            method="abstract_only",
            full_text_available=False,
        )
