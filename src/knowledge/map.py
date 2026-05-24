"""In-memory Knowledge Map with CRUD operations and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.knowledge.models import (
    Evidence,
    EvidenceType,
    Gap,
    KnowledgeNode,
    PaperMeta,
)


def summarize_knowledge_map(
    nodes: list[dict],
    max_topics: int = 40,
    max_claims: int = 25,
    claim_max_len: int = 150,
    dedup_topics: bool = True,
    topics_label: str = "Topics",
    claims_label: str = "Key findings",
) -> str:
    """Create a concise text summary of knowledge nodes for LLM prompts.

    Returns a string like:
        Topics (5): topic1, topic2, ...
        Key findings:
        - claim one
        - claim two
    """
    if dedup_topics:
        topics = list(dict.fromkeys(n.get("topic", "") for n in nodes if n.get("topic")))
    else:
        topics = [n.get("topic", "Unknown") for n in nodes if n.get("topic")]

    claims = [n.get("claim", "")[:claim_max_len] for n in nodes if n.get("claim")]

    lines = [f"{topics_label} ({len(topics)}): {', '.join(topics[:max_topics])}", ""]
    lines.append(f"{claims_label}:")
    for c in claims[:max_claims]:
        lines.append(f"- {c}")

    return "\n".join(lines)


class KnowledgeMap:
    """In-memory knowledge map with provenance tracking."""

    def __init__(self):
        self._nodes: dict[str, KnowledgeNode] = {}

    # --- CRUD ---

    def add_node(self, node: KnowledgeNode) -> None:
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)

    def list_nodes(self) -> list[KnowledgeNode]:
        return list(self._nodes.values())

    def node_count(self) -> int:
        return len(self._nodes)

    # --- Evidence ---

    def add_evidence(
        self,
        node_id: str,
        paper: PaperMeta,
        section: str = "",
        quote: str = "",
        evidence_type: EvidenceType = EvidenceType.ORIGINAL_CLAIM,
    ) -> Evidence | None:
        node = self._nodes.get(node_id)
        if node is None:
            return None
        evidence = Evidence(
            paper=paper,
            section=section,
            quote=quote,
            evidence_type=evidence_type,
        )
        node.evidence.append(evidence)
        return evidence

    def get_evidence_for_paper(self, paper_id: str) -> list[tuple[str, Evidence]]:
        """Return all evidence entries from a specific paper as (node_id, evidence) pairs."""
        results = []
        for node in self._nodes.values():
            for ev in node.evidence:
                if ev.paper.paper_id == paper_id:
                    results.append((node.id, ev))
        return results

    def has_evidence_from(self, paper_id: str) -> bool:
        """Check if any evidence from this paper has been extracted."""
        for node in self._nodes.values():
            for ev in node.evidence:
                if ev.paper.paper_id == paper_id:
                    return True
        return False

    # --- Relationships ---

    def link_nodes(self, node_id_a: str, node_id_b: str) -> None:
        """Create a bidirectional link between two nodes."""
        node_a = self._nodes.get(node_id_a)
        node_b = self._nodes.get(node_id_b)
        if node_a and node_id_b not in node_a.related_nodes:
            node_a.related_nodes.append(node_id_b)
        if node_b and node_id_a not in node_b.related_nodes:
            node_b.related_nodes.append(node_id_a)

    def find_by_topic(self, topic: str) -> list[KnowledgeNode]:
        """Find nodes whose topic contains the given substring (case-insensitive)."""
        topic_lower = topic.lower()
        return [n for n in self._nodes.values() if topic_lower in n.topic.lower()]

    # --- Serialization ---

    def to_dict(self) -> dict:
        return {
            "nodes": [n.model_dump() for n in self._nodes.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeMap":
        km = cls()
        for node_data in data.get("nodes", []):
            node = KnowledgeNode(**node_data)
            km.add_node(node)
        return km

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeMap":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # --- Summary ---

    def summary(self) -> dict:
        papers = set()
        for node in self._nodes.values():
            for ev in node.evidence:
                papers.add(ev.paper.paper_id)
        return {
            "node_count": len(self._nodes),
            "unique_papers": len(papers),
            "nodes_by_confidence": {
                "high": sum(1 for n in self._nodes.values() if n.confidence >= 0.7),
                "medium": sum(1 for n in self._nodes.values() if 0.3 <= n.confidence < 0.7),
                "low": sum(1 for n in self._nodes.values() if n.confidence < 0.3),
            },
        }
