"""Node: update_knowledge_map — extract claims and update the knowledge map."""

from __future__ import annotations

import json
import logging

from src.agent.state import AgentState
from src.config.settings import get_settings
from src.knowledge.models import Evidence, EvidenceType, KnowledgeNode, PaperMeta
from src.knowledge.map import KnowledgeMap
from src.llm.client import get_llm_client
from src.llm.prompts import get_prompt_manager
from src.utils.text_selector import select_text

logger = logging.getLogger(__name__)


async def update_knowledge_map_node(state: AgentState) -> AgentState:
    """Extract claims from newly read papers and update the knowledge map.

    Reads: papers_read, paper_index, knowledge_nodes
    Writes: knowledge_nodes
    """
    papers_read = state.get("papers_read", {})
    paper_index = state.get("paper_index", {})
    km_data = state.get("knowledge_nodes", [])

    km = KnowledgeMap()
    for node_data in km_data:
        try:
            node = KnowledgeNode(**node_data)
            km.add_node(node)
        except Exception as e:
            logger.warning(f"Failed to deserialize node: {e}")

    # Determine which papers are new (not yet in knowledge map)
    new_papers = {
        pid: text
        for pid, text in papers_read.items()
        if not km.has_evidence_from(pid)
    }

    if not new_papers:
        logger.info("[update_knowledge_map] No new papers to process")
        state["knowledge_nodes"] = [n.model_dump() for n in km.list_nodes()]
        return state

    settings = get_settings()
    llm = get_llm_client()
    pm = get_prompt_manager()

    existing_topics = [n.topic for n in km.list_nodes()]
    existing_topics_str = ", ".join(existing_topics) if existing_topics else "None"

    for paper_id, text in new_papers.items():
        paper = paper_index.get(paper_id, {})
        if not text:
            continue

        try:
            response = await llm.complete_json(
                system_prompt="You extract structured claims from academic papers.",
                user_prompt=pm.get(
                    "extract_claims",
                    title=paper.get("title", "Unknown"),
                    authors=", ".join(paper.get("authors", [])),
                    year=str(paper.get("year", "unknown")),
                    venue=paper.get("venue", "unknown"),
                    existing_topics=existing_topics_str,
                    paper_text=select_text(text, max_chars=8000),
                ),
            )
            claims = response if isinstance(response, list) else response.get("claims", [])

            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                _process_claim(claim, paper, km)

                # Update existing_topics for subsequent papers
                topic = claim.get("topic", "")
                if topic and topic not in existing_topics:
                    existing_topics.append(topic)

        except Exception as e:
            logger.warning(f"Failed to extract claims from {paper_id}: {e}")
            state.setdefault("errors", []).append({
                "node": "update_knowledge_map",
                "paper_id": paper_id,
                "error": str(e),
            })

    state["knowledge_nodes"] = [n.model_dump() for n in km.list_nodes()]
    logger.info(
        f"[update_knowledge_map] {len(new_papers)} papers → "
        f"{km.node_count()} total knowledge nodes"
    )
    return state


def _process_claim(
    claim: dict,
    paper: dict,
    km: KnowledgeMap,
) -> None:
    """Process a single extracted claim and add/update the knowledge map."""
    topic = claim.get("topic", "")
    claim_text = claim.get("claim", "")
    if not topic or not claim_text:
        return

    paper_meta = _dict_to_paper_meta(paper)
    evidence_type = EvidenceType.ORIGINAL_CLAIM
    if claim.get("evidence_type") in ("supporting", "contradicting"):
        evidence_type = EvidenceType(claim["evidence_type"])

    evidence = Evidence(
        paper=paper_meta,
        section=claim.get("section", ""),
        quote=claim.get("quote", ""),
        evidence_type=evidence_type,
    )

    # Find existing node by topic
    related = km.find_by_topic(topic)

    if related:
        node = related[0]
        node.evidence.append(evidence)
        if claim.get("related_to") and claim["related_to"] in [n.topic for n in km.list_nodes()]:
            related_to = [n for n in km.list_nodes() if n.topic == claim["related_to"]]
            if related_to:
                km.link_nodes(node.id, related_to[0].id)
    else:
        node_id = f"node_{km.node_count() + 1:03d}"
        node = KnowledgeNode(
            id=node_id,
            topic=topic,
            claim=claim_text,
            evidence=[evidence],
            confidence=claim.get("confidence", 0.5),
        )
        km.add_node(node)


def _dict_to_paper_meta(paper: dict) -> PaperMeta:
    return PaperMeta(
        paper_id=paper.get("paper_id", ""),
        title=paper.get("title", ""),
        authors=paper.get("authors", []),
        year=paper.get("year"),
        url=paper.get("url", ""),
        venue=paper.get("venue", ""),
        citation_count=paper.get("citation_count", 0),
        abstract=paper.get("abstract", ""),
        full_text_available=paper.get("full_text_available", False),
        source=paper.get("source", ""),
    )
