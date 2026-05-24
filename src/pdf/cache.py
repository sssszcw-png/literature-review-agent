"""File-based cache for parsed papers."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PaperCache:
    """Manages cached parsed-paper markdown files.

    Cache location: cache/papers/{paper_id}.md
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, paper_id: str) -> Path:
        safe_id = paper_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_id}.md"

    def get(self, paper_id: str) -> str | None:
        """Return cached parsed text, or None if not cached."""
        path = self._path(paper_id)
        if path.exists():
            logger.info(f"Cache hit: {paper_id}")
            return path.read_text(encoding="utf-8")
        logger.info(f"Cache miss: {paper_id}")
        return None

    def put(self, paper_id: str, text: str) -> None:
        """Store parsed text in cache."""
        path = self._path(paper_id)
        path.write_text(text, encoding="utf-8")
        logger.info(f"Cached: {paper_id} → {path}")

    def has(self, paper_id: str) -> bool:
        return self._path(paper_id).exists()

    def clear(self) -> int:
        """Remove all cached papers. Returns count of files deleted."""
        count = 0
        for f in self.cache_dir.glob("*.md"):
            f.unlink()
            count += 1
        return count

    def stats(self) -> dict:
        """Return cache statistics."""
        files = list(self.cache_dir.glob("*.md"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "cached_papers": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }
