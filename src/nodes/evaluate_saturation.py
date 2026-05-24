"""Node: evaluate_saturation — LLM-as-a-Judge scoring for knowledge saturation."""

from __future__ import annotations

import json
import logging

from src.agent.state import AgentState
from src.config.settings import get_settings
from src.knowledge.models import Gap, KnowledgeNode, SaturationScores
from src.llm.client import get_llm_client
from src.llm.prompts import get_prompt_manager

logger = logging.getLogger(__name__)


async def evaluate_saturation_node(state: AgentState) -> AgentState:
    """Evaluate knowledge saturation for each gap using LLM-as-a-Judge.

    Reads: gaps, knowledge_nodes
    Writes: gaps (updated saturation), last_saturation_scores, consecutive_no_improvement
    """
    gaps_data = state.get("gaps", [])
    km_data = state.get("knowledge_nodes", [])
    last_scores = state.get("last_saturation_scores", {})

    if not gaps_data:
        logger.info("[evaluate_saturation] No gaps to evaluate")
        return state

    # Deserialize
    gaps = [Gap(**g) if isinstance(g, dict) else g for g in gaps_data]
    km_nodes = [KnowledgeNode(**n) if isinstance(n, dict) else n for n in km_data]

    settings = get_settings()
    llm = get_llm_client()
    pm = get_prompt_manager()

    current_scores = {}
    improved = False

    for gap in gaps:
        if gap.saturation >= settings.saturation_threshold:
            current_scores[gap.description] = gap.saturation
            continue

        # Find related knowledge nodes for this gap
        related_nodes = _find_related_nodes(gap, km_nodes)
        if not related_nodes:
            current_scores[gap.description] = 0.0
            continue

        nodes_json = json.dumps(
            [{
                "topic": n.topic,
                "claim": n.claim,
                "confidence": n.confidence,
                "evidence_count": len(n.evidence),
                "papers": list(set(e.paper.title for e in n.evidence)),
                "venues": list(set(e.paper.venue for e in n.evidence)),
                "citation_counts": [e.paper.citation_count for e in n.evidence],
            } for n in related_nodes],
            indent=2,
            ensure_ascii=False,
        )

        try:
            scores_dict = await llm.complete_json(
                system_prompt="You evaluate research saturation for academic gaps.",
                user_prompt=pm.get(
                    "evaluate_saturation",
                    gap_description=gap.description,
                    knowledge_nodes_json=nodes_json,
                ),
            )
            scores = SaturationScores(
                coverage=scores_dict.get("coverage", 0.0),
                source_quality=scores_dict.get("source_quality", 0.0),
                consensus=scores_dict.get("consensus", 0.0),
            )
        except Exception as e:
            logger.warning(f"Saturation evaluation failed: {e}")
            scores = SaturationScores(coverage=0.0, source_quality=0.0, consensus=0.0)

        gap.saturation = scores.saturation
        gap.saturation_detail = {
            "coverage": scores.coverage,
            "source_quality": scores.source_quality,
            "consensus": scores.consensus,
        }
        current_scores[gap.description] = gap.saturation

        logger.info(
            f"Saturation for gap '{gap.description[:50]}...': "
            f"coverage={scores.coverage:.2f}, quality={scores.source_quality:.2f}, "
            f"consensus={scores.consensus:.2f} → saturation={scores.saturation:.2f}"
        )

    # Check improvement
    if last_scores:
        improved = any(
            current_scores.get(k, 0) > last_scores.get(k, 0)
            for k in current_scores
        )

    consecutive = state.get("consecutive_no_improvement", 0)
    if not improved and last_scores:
        consecutive += 1
    else:
        consecutive = 0

    state["gaps"] = [g.model_dump() for g in gaps]
    state["last_saturation_scores"] = current_scores
    state["consecutive_no_improvement"] = consecutive

    return state


def _find_related_nodes(gap: Gap, nodes: list[KnowledgeNode]) -> list[KnowledgeNode]:
    """Find knowledge nodes related to a gap by keyword overlap."""
    gap_words = set(gap.description.lower().split())
    related = []
    for node in nodes:
        node_words = set((node.topic + " " + node.claim).lower().split())
        overlap = len(gap_words & node_words)
        if overlap >= 2:
            related.append(node)
    return related
