"""Docling PDF parser wrapper for academic papers."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DoclingParser:
    """Wraps Docling for ML-driven academic PDF parsing.

    Docling handles: dual-column layout, formulas, tables, figures.
    """

    async def parse(self, pdf_path: Path) -> str | None:
        """Parse a PDF file and return Markdown text, or None on failure."""
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            # Docling conversion is CPU-bound, run in a thread
            import asyncio
            result = await asyncio.to_thread(
                converter.convert, str(pdf_path)
            )
            if result and result.document:
                markdown = result.document.export_to_markdown()
                return markdown
            return None
        except ImportError:
            logger.warning("Docling not installed; skipping Docling parser")
            return None
        except Exception as e:
            logger.warning(f"Docling parse failed for {pdf_path.name}: {e}")
            return None
