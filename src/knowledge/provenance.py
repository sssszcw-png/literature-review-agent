"""Provenance validation — every claim must be traceable to a paper, section, quote."""

from src.knowledge.models import Evidence, KnowledgeNode


class ProvenanceError(Exception):
    """A provenance validation failure."""

    pass


def validate_evidence(evidence: Evidence) -> None:
    """Raise ProvenanceError if evidence is missing required provenance fields."""
    if not evidence.paper:
        raise ProvenanceError("Evidence must reference a PaperMeta")
    if not evidence.paper.paper_id:
        raise ProvenanceError("Evidence paper must have a paper_id")
    if not evidence.paper.title:
        raise ProvenanceError("Evidence paper must have a title")
    if not evidence.quote:
        raise ProvenanceError("Evidence must include a verbatim quote")


def validate_node(node: KnowledgeNode) -> list[str]:
    """Validate all evidence on a KnowledgeNode. Returns list of error messages."""
    errors = []
    if not node.evidence:
        errors.append(f"Node '{node.id}': no evidence attached")
    for i, ev in enumerate(node.evidence):
        try:
            validate_evidence(ev)
        except ProvenanceError as e:
            errors.append(f"Node '{node.id}', evidence [{i}]: {e}")
    return errors


def validate_knowledge_map(nodes: list[KnowledgeNode]) -> list[str]:
    """Validate all nodes in the knowledge map. Returns list of all errors found."""
    all_errors = []
    for node in nodes:
        all_errors.extend(validate_node(node))
    return all_errors


def has_full_provenance(node: KnowledgeNode) -> bool:
    """Check if a node has complete provenance (all evidence has quotes and paper refs)."""
    if not node.evidence:
        return False
    for ev in node.evidence:
        if not ev.paper or not ev.paper.paper_id or not ev.quote:
            return False
    return True
