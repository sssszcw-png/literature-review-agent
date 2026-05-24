"""PDF downloader using aiohttp."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import aiohttp

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class PDFDownloader:
    """Downloads PDF files from URLs with size limits and timeout."""

    def __init__(self, settings: Settings):
        self.download_dir = Path(settings.pdf_download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = settings.pdf_parse_timeout
        self.max_size_bytes = settings.pdf_max_size_mb * 1024 * 1024

    async def download(self, url: str, paper_id: str) -> Path | None:
        """Download a PDF and save to disk. Returns the file path or None on failure."""

        # Use a sensible filename
        safe_id = paper_id.replace("/", "_").replace("\\", "_")
        output_path = self.download_dir / f"{safe_id}.pdf"

        # Skip if already downloaded
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"PDF already downloaded: {paper_id}")
            return output_path

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    allow_redirects=True,
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"PDF download failed for {paper_id}: HTTP {resp.status}"
                        )
                        return None

                    content_type = resp.headers.get("Content-Type", "")
                    if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                        logger.warning(
                            f"PDF download unexpected content type: {content_type}"
                        )

                    # Stream and check size
                    data = bytearray()
                    async for chunk in resp.content.iter_chunked(8192):
                        data.extend(chunk)
                        if len(data) > self.max_size_bytes:
                            logger.warning(f"PDF too large ({len(data)} bytes): {paper_id}")
                            return None

                    output_path.write_bytes(bytes(data))
                    logger.info(f"Downloaded PDF: {paper_id} ({len(data)} bytes)")
                    return output_path

        except asyncio.TimeoutError:
            logger.warning(f"PDF download timed out: {paper_id}")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"PDF download failed for {paper_id}: {e}")
            return None

    def cleanup(self, paper_id: str) -> None:
        """Delete a downloaded PDF to free disk space."""
        safe_id = paper_id.replace("/", "_").replace("\\", "_")
        path = self.download_dir / f"{safe_id}.pdf"
        if path.exists():
            path.unlink()
