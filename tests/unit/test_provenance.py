"""Tests for provenance validation."""

import pytest
from src.knowledge.models import Evidence, EvidenceType, KnowledgeNode, PaperMeta
from src.knowledge.provenance import (
    ProvenanceError,
    has_full_provenance,
    validate_evidence,
    validate_knowledge_map,
    validate_node,
)


class TestValidateEvidence:
    def test_valid_evidence(self, sample_evidence):
        validate_evidence(sample_evidence)  # Should not raise

    def test_missing_paper(self):
        # Pydantic prevents creating Evidence with paper=None at the model level
        # The provenance validation handles empty paper_id/quote within a valid PaperMeta
        with pytest.raises(Exception):  # Pydantic ValidationError
            Evidence(paper=None, quote="test")

    def test_missing_paper_id(self):
        ev = Evidence(
            paper=PaperMeta(paper_id="", title="Test"),
            quote="test",
        )
        with pytest.raises(ProvenanceError):
            validate_evidence(ev)

    def test_missing_quote(self, sample_paper_meta):
        ev = Evidence(paper=sample_paper_meta, quote="")
        with pytest.raises(ProvenanceError):
            validate_evidence(ev)

    def test_missing_title(self):
        ev = Evidence(
            paper=PaperMeta(paper_id="ss:123", title=""),
            quote="test",
        )
        with pytest.raises(ProvenanceError):
            validate_evidence(ev)


class TestValidateNode:
    def test_valid_node(self, sample_knowledge_node):
        errors = validate_node(sample_knowledge_node)
        assert len(errors) == 0

    def test_no_evidence(self):
        node = KnowledgeNode(id="n1", topic="X", claim="Y", evidence=[])
        errors = validate_node(node)
        assert len(errors) >= 1
        assert "no evidence" in errors[0].lower()

    def test_bad_evidence(self, sample_paper_meta):
        ev = Evidence(paper=sample_paper_meta, quote="")  # missing quote
        node = KnowledgeNode(id="n1", topic="X", claim="Y", evidence=[ev])
        errors = validate_node(node)
        assert len(errors) >= 1


class TestValidateKnowledgeMap:
    def test_all_valid(self, sample_knowledge_nodes):
        errors = validate_knowledge_map(sample_knowledge_nodes)
        assert len(errors) == 0

    def test_mixed_valid_invalid(self, sample_knowledge_nodes, sample_paper_meta):
        bad_ev = Evidence(paper=sample_paper_meta, quote="")
        bad_node = KnowledgeNode(id="bad", topic="X", claim="Y", evidence=[bad_ev])
        nodes = sample_knowledge_nodes + [bad_node]
        errors = validate_knowledge_map(nodes)
        assert len(errors) >= 1


class TestHasFullProvenance:
    def test_full_provenance(self, sample_knowledge_node):
        assert has_full_provenance(sample_knowledge_node) is True

    def test_no_evidence(self):
        node = KnowledgeNode(id="n1", topic="X", claim="Y", evidence=[])
        assert has_full_provenance(node) is False

    def test_missing_quote(self, sample_paper_meta):
        ev = Evidence(paper=sample_paper_meta, quote="")
        node = KnowledgeNode(id="n1", topic="X", claim="Y", evidence=[ev])
        assert has_full_provenance(node) is False
