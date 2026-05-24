"""Marker PDF parser fallback wrapper."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MarkerParser:
    """Wraps Marker as a fallback PDF-to-Markdown parser."""

    async def parse(self, pdf_path: Path) -> str | None:
        """Parse a PDF file using Marker and return Markdown text, or None on failure."""
        try:
            from marker.converters.pdf import PdfConverter

            import asyncio

            converter = PdfConverter()
            result = await asyncio.to_thread(
                converter.convert, str(pdf_path)
            )
            if result:
                return result
            return None
        except ImportError:
            logger.warning("Marker not installed; skipping Marker parser")
            return None
        except Exception as e:
            logger.warning(f"Marker parse failed for {pdf_path.name}: {e}")
            return None
