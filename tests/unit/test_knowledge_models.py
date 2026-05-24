"""Tests for knowledge domain models."""

import pytest
from src.knowledge.models import (
    Evidence,
    EvidenceType,
    Gap,
    GapSeverity,
    KnowledgeNode,
    Outline,
    OutlineItem,
    PaperMeta,
    SaturationScores,
)


class TestPaperMeta:
    def test_create_minimal(self):
        p = PaperMeta(paper_id="ss:123", title="Test Paper")
        assert p.paper_id == "ss:123"
        assert p.title == "Test Paper"
        assert p.authors == []
        assert p.year is None

    def test_create_full(self, sample_paper_meta):
        assert sample_paper_meta.citation_count == 100000
        assert len(sample_paper_meta.authors) == 3
        assert sample_paper_meta.source == "semantic_scholar"

    def test_serialize_deserialize(self, sample_paper_meta):
        data = sample_paper_meta.model_dump()
        restored = PaperMeta(**data)
        assert restored.title == sample_paper_meta.title
        assert restored.paper_id == sample_paper_meta.paper_id


class TestEvidence:
    def test_create(self, sample_paper_meta):
        ev = Evidence(
            paper=sample_paper_meta,
            section="Section 3",
            quote="The transformer uses...",
            evidence_type=EvidenceType.ORIGINAL_CLAIM,
        )
        assert ev.paper.paper_id == "ss:abc123"
        assert ev.evidence_type == EvidenceType.ORIGINAL_CLAIM

    def test_supporting_evidence(self, sample_paper_meta):
        ev = Evidence(
            paper=sample_paper_meta,
            quote="Our results confirm...",
            evidence_type=EvidenceType.SUPPORTING,
        )
        assert ev.evidence_type == EvidenceType.SUPPORTING

    def test_contradicting_evidence(self, sample_paper_meta):
        ev = Evidence(
            paper=sample_paper_meta,
            quote="Contrary to prior work...",
            evidence_type=EvidenceType.CONTRADICTING,
        )
        assert ev.evidence_type == EvidenceType.CONTRADICTING


class TestKnowledgeNode:
    def test_create(self, sample_evidence):
        node = KnowledgeNode(
            id="n1",
            topic="Self-Attention",
            claim="Attention is effective",
            evidence=[sample_evidence],
        )
        assert node.id == "n1"
        assert node.confidence == 0.5  # default

    def test_confidence_bounds(self, sample_evidence):
        with pytest.raises(Exception):
            KnowledgeNode(
                id="n1",
                topic="Test",
                claim="Test",
                evidence=[sample_evidence],
                confidence=1.5,
            )

    def test_related_nodes(self, sample_evidence):
        node = KnowledgeNode(
            id="n1",
            topic="Test",
            claim="Test",
            evidence=[sample_evidence],
            related_nodes=["n2", "n3"],
        )
        assert len(node.related_nodes) == 2

    def test_serialize_round_trip(self, sample_knowledge_node):
        data = sample_knowledge_node.model_dump()
        restored = KnowledgeNode(**data)
        assert restored.id == sample_knowledge_node.id
        assert restored.topic == sample_knowledge_node.topic
        assert len(restored.evidence) == 1


class TestGap:
    def test_create(self):
        gap = Gap(description="Missing evidence on X", severity=GapSeverity.CRITICAL)
        assert gap.severity == GapSeverity.CRITICAL
        assert gap.saturation == 0.0

    def test_saturation_bounds(self):
        with pytest.raises(Exception):
            Gap(description="Test", saturation=1.5)

    def test_with_scores(self):
        gap = Gap(
            description="Gap in coverage",
            severity=GapSeverity.IMPORTANT,
            saturation=0.65,
            saturation_detail={"coverage": 0.6, "source_quality": 0.7, "consensus": 0.65},
        )
        assert abs(gap.saturation - 0.65) < 0.01


class TestSaturationScores:
    def test_average(self):
        scores = SaturationScores(coverage=0.8, source_quality=0.6, consensus=0.7)
        expected = (0.8 + 0.6 + 0.7) / 3.0
        assert abs(scores.saturation - expected) < 0.01

    def test_zero(self):
        scores = SaturationScores(coverage=0.0, source_quality=0.0, consensus=0.0)
        assert scores.saturation == 0.0

    def test_perfect(self):
        scores = SaturationScores(coverage=1.0, source_quality=1.0, consensus=1.0)
        assert scores.saturation == 1.0


class TestOutline:
    def test_create(self):
        chapters = [
            OutlineItem(title="Introduction", topics=["Background"], description="Overview"),
            OutlineItem(title="Conclusion", topics=["Summary"], description="Synthesis"),
        ]
        outline = Outline(title="Review", chapters=chapters)
        assert len(outline.chapters) == 2
        assert outline.chapters[0].title == "Introduction"
