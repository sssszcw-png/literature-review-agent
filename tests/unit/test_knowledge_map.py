"""Tests for KnowledgeMap CRUD operations."""

import json
import tempfile
from pathlib import Path

from src.knowledge.map import KnowledgeMap
from src.knowledge.models import (
    Evidence,
    EvidenceType,
    KnowledgeNode,
    PaperMeta,
)


class TestKnowledgeMapCRUD:
    def test_add_and_get(self):
        km = KnowledgeMap()
        node = KnowledgeNode(id="n1", topic="X", claim="X is Y")
        km.add_node(node)
        assert km.get_node("n1") is not None
        assert km.get_node("n1").topic == "X"

    def test_get_missing(self):
        km = KnowledgeMap()
        assert km.get_node("nonexistent") is None

    def test_remove_node(self):
        km = KnowledgeMap()
        node = KnowledgeNode(id="n1", topic="X", claim="Y")
        km.add_node(node)
        km.remove_node("n1")
        assert km.get_node("n1") is None

    def test_remove_missing_no_error(self):
        km = KnowledgeMap()
        km.remove_node("nonexistent")

    def test_node_count(self, sample_knowledge_nodes):
        km = KnowledgeMap()
        for node in sample_knowledge_nodes:
            km.add_node(node)
        assert km.node_count() == len(sample_knowledge_nodes)

    def test_list_nodes(self, sample_knowledge_nodes):
        km = KnowledgeMap()
        for node in sample_knowledge_nodes:
            km.add_node(node)
        nodes = km.list_nodes()
        assert len(nodes) == len(sample_knowledge_nodes)


class TestKnowledgeMapEvidence:
    def test_add_evidence(self, sample_knowledge_map, sample_paper_meta):
        km = sample_knowledge_map
        nodes = km.list_nodes()
        assert len(nodes) > 0

        ev = km.add_evidence(
            node_id=nodes[0].id,
            paper=sample_paper_meta,
            section="Section 5",
            quote="New finding...",
            evidence_type=EvidenceType.SUPPORTING,
        )
        assert ev is not None
        updated = km.get_node(nodes[0].id)
        assert len(updated.evidence) == 2  # original + new

    def test_add_evidence_missing_node(self, sample_paper_meta):
        km = KnowledgeMap()
        ev = km.add_evidence("no_such_node", sample_paper_meta)
        assert ev is None

    def test_has_evidence_from(self, sample_knowledge_map, sample_papers):
        km = sample_knowledge_map
        assert km.has_evidence_from(sample_papers[0].paper_id)
        assert not km.has_evidence_from("unknown_paper")

    def test_get_evidence_for_paper(self, sample_knowledge_map, sample_papers):
        km = sample_knowledge_map
        ev_pairs = km.get_evidence_for_paper(sample_papers[0].paper_id)
        assert len(ev_pairs) >= 1
        node_id, evidence = ev_pairs[0]
        assert evidence.paper.paper_id == sample_papers[0].paper_id


class TestKnowledgeMapRelationships:
    def test_link_nodes(self):
        km = KnowledgeMap()
        km.add_node(KnowledgeNode(id="n1", topic="A", claim="X"))
        km.add_node(KnowledgeNode(id="n2", topic="B", claim="Y"))
        km.link_nodes("n1", "n2")

        assert "n2" in km.get_node("n1").related_nodes
        assert "n1" in km.get_node("n2").related_nodes

    def test_link_missing_node_no_error(self):
        km = KnowledgeMap()
        km.add_node(KnowledgeNode(id="n1", topic="A", claim="X"))
        km.link_nodes("n1", "n2")  # n2 doesn't exist


class TestKnowledgeMapFind:
    def test_find_by_topic(self, sample_knowledge_map, sample_knowledge_nodes):
        km = sample_knowledge_map
        first_topic = sample_knowledge_nodes[0].topic
        found = km.find_by_topic(first_topic)
        assert len(found) >= 1

    def test_find_by_topic_case_insensitive(self, sample_knowledge_map, sample_knowledge_nodes):
        km = sample_knowledge_map
        first_topic = sample_knowledge_nodes[0].topic.lower()
        found = km.find_by_topic(first_topic)
        assert len(found) >= 1

    def test_find_no_match(self):
        km = KnowledgeMap()
        assert km.find_by_topic("nothing_matches_this") == []


class TestKnowledgeMapSerialization:
    def test_to_dict_and_back(self, sample_knowledge_map):
        data = sample_knowledge_map.to_dict()
        assert "nodes" in data
        restored = KnowledgeMap.from_dict(data)
        assert restored.node_count() == sample_knowledge_map.node_count()

    def test_save_and_load(self, sample_knowledge_map):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            sample_knowledge_map.save(temp_path)
            loaded = KnowledgeMap.load(temp_path)
            assert loaded.node_count() == sample_knowledge_map.node_count()
        finally:
            temp_path.unlink(missing_ok=True)


class TestKnowledgeMapSummary:
    def test_summary_empty(self):
        km = KnowledgeMap()
        s = km.summary()
        assert s["node_count"] == 0
        assert s["unique_papers"] == 0

    def test_summary_with_data(self, sample_knowledge_map, sample_knowledge_nodes):
        km = sample_knowledge_map
        s = km.summary()
        assert s["node_count"] == len(sample_knowledge_nodes)
        assert s["unique_papers"] >= 1
