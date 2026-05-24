"""Tests for chapter stitching."""

from src.report.stitch import build_citation_index, stitch_chapters
from src.knowledge.models import Evidence, EvidenceType, KnowledgeNode


class TestBuildCitationIndex:
    def test_empty_nodes(self):
        index = build_citation_index([])
        assert index == {}

    def test_single_paper(self, sample_knowledge_nodes):
        index = build_citation_index(sample_knowledge_nodes)
        # Each node has evidence from a different paper
        assert len(index) == len(sample_knowledge_nodes)

    def test_duplicate_papers(self, sample_knowledge_nodes):
        # Add a second evidence from the same paper
        node = sample_knowledge_nodes[0]
        existing_ev = node.evidence[0]
        new_ev = Evidence(
            paper=existing_ev.paper,
            section="Section 7",
            quote="Another quote",
            evidence_type=EvidenceType.SUPPORTING,
        )
        node.evidence.append(new_ev)

        index = build_citation_index(sample_knowledge_nodes)
        # Should still have unique paper_ids count
        paper_ids = set(ev.paper.paper_id for n in sample_knowledge_nodes for ev in n.evidence)
        assert len(index) == len(paper_ids)


class TestStitchChapters:
    def test_basic_stitch(self, sample_outline, sample_citation_index):
        chapters = {
            "Introduction": "This is the introduction [1].",
            "Self-Attention Mechanisms": "Core attention architecture discussion [1].",
            "Transformer Variants": "Variants of the transformer [2].",
            "Conclusion": "In conclusion, transformers are important [1][2].",
        }
        draft = stitch_chapters(
            {"title": sample_outline.title, "chapters": [c.model_dump() for c in sample_outline.chapters]},
            chapters,
            sample_citation_index,
        )
        assert "Introduction" in draft
        assert "Self-Attention Mechanisms" in draft
        assert "## References" in draft
        assert "[1]" in draft
        assert "Attention Is All You Need" in draft

    def test_missing_chapter_content(self, sample_outline, sample_citation_index):
        chapters = {}  # No chapters generated
        draft = stitch_chapters(
            {"title": sample_outline.title, "chapters": [c.model_dump() for c in sample_outline.chapters]},
            chapters,
            sample_citation_index,
        )
        for ch in sample_outline.chapters:
            assert ch.title in draft

    def test_references_formatted(self, sample_outline, sample_citation_index):
        chapters = {ch.title: "Content" for ch in sample_outline.chapters}
        draft = stitch_chapters(
            {"title": sample_outline.title, "chapters": [c.model_dump() for c in sample_outline.chapters]},
            chapters,
            sample_citation_index,
        )
        ref_section = draft.split("## References")[1]
        assert "Attention Is All You Need" in ref_section
        assert "BERT" in ref_section
