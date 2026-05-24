"""Search data models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single paper returned from search."""

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    url: str = ""
    venue: str = ""
    citation_count: int = 0
    abstract: str = ""
    source: str = ""  # "semantic_scholar" or "arxiv"
    pdf_url: str = ""
    doi: str = ""

    def first_author_surname(self) -> str:
        if self.authors:
            return self.authors[0].split()[-1] if self.authors[0].split() else ""
        return ""
