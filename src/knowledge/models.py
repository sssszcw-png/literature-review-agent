"""Core domain models for the knowledge map."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    ORIGINAL_CLAIM = "original_claim"
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"


class GapSeverity(str, Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    NICE_TO_HAVE = "nice_to_have"


class PaperMeta(BaseModel):
    """Metadata for an academic paper."""

    paper_id: str = Field(description="Unique ID (SS Corpus ID or arXiv ID)")
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    url: str = ""
    venue: str = ""
    citation_count: int = 0
    abstract: str = ""
    full_text_available: bool = False
    source: str = ""  # "semantic_scholar" or "arxiv"


class Evidence(BaseModel):
    """A piece of evidence linking a claim to a specific paper location."""

    paper: PaperMeta
    section: str = Field(default="", description="Section or page reference")
    quote: str = Field(default="", description="Verbatim quote from the paper")
    evidence_type: EvidenceType = EvidenceType.ORIGINAL_CLAIM


class KnowledgeNode(BaseModel):
    """A node in the knowledge map representing a claim with supporting evidence."""

    id: str
    topic: str
    claim: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    related_nodes: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class Gap(BaseModel):
    """A knowledge gap that needs further investigation."""

    description: str
    severity: GapSeverity = GapSeverity.IMPORTANT
    saturation: float = Field(default=0.0, ge=0.0, le=1.0)
    saturation_detail: dict = Field(
        default_factory=dict,
        description="Breakdown: {coverage, source_quality, consensus}",
    )


class SaturationScores(BaseModel):
    """LLM-as-a-Judge saturation evaluation output."""

    coverage: float = Field(ge=0.0, le=1.0, description="How thoroughly sources address the gap")
    source_quality: float = Field(ge=0.0, le=1.0, description="Weighted by venue, citations, recency")
    consensus: float = Field(ge=0.0, le=1.0, description="Agreement (high) vs contradiction (low)")

    @property
    def saturation(self) -> float:
        return (self.coverage + self.source_quality + self.consensus) / 3.0


class OutlineItem(BaseModel):
    """A single chapter/section in the report outline."""

    title: str
    topics: list[str] = Field(default_factory=list)
    description: str = ""


class Outline(BaseModel):
    """Report outline with chapters."""

    title: str
    chapters: list[OutlineItem] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
