"""Node: detect_gaps — identify knowledge gaps and classify severity."""

from __future__ import annotations

import json
import logging

from src.agent.state import AgentState
from src.config.constants import PHASE_BROAD
from src.config.settings import get_settings
from src.knowledge.map import summarize_knowledge_map
from src.knowledge.models import Gap, GapSeverity, KnowledgeNode
from src.llm.client import get_llm_client
from src.llm.prompts import get_prompt_manager

logger = logging.getLogger(__name__)


async def detect_gaps_node(state: AgentState) -> AgentState:
    """Analyze the knowledge map and detect coverage gaps.

    In Round 1 (broad): runs LLM to detect gaps from scratch.
    In Round 2+ (deep_dive): preserves existing gaps — saturation is
    updated by evaluate_saturation. Only re-detects if no gaps exist yet.

    Reads: knowledge_nodes, research_question, current_round, phase
    Writes: gaps (with severity)
    """
    km_data = state.get("knowledge_nodes", [])
    question = state.get("research_question", "")
    phase = state.get("phase", PHASE_BROAD)
    existing_gaps = state.get("gaps", [])

    if not km_data:
        gap = Gap(
            description=f"No papers found yet for: {question}",
            severity=GapSeverity.CRITICAL,
        )
        state["gaps"] = [gap.model_dump()]
        return state

    # If gaps already exist from a previous round, don't regenerate.
    # Saturation scores flow through evaluate_saturation → state update.
    if existing_gaps and phase != PHASE_BROAD:
        logger.info(f"[detect_gaps] Preserving {len(existing_gaps)} existing gaps")
        return state

    return await _detect_gaps_fresh(state, km_data, question)


async def _detect_gaps_fresh(
    state: AgentState,
    km_data: list[dict],
    question: str,
) -> AgentState:
    """Run LLM to detect knowledge gaps from the current knowledge map."""
    settings = get_settings()
    llm = get_llm_client()
    pm = get_prompt_manager()

    km_summary = summarize_knowledge_map(
        km_data, max_topics=30, max_claims=20, claim_max_len=200,
        topics_label="Topics covered", claims_label="Key claims",
    )

    try:
        gaps_json = await llm.complete_json(
            system_prompt="You identify research gaps in academic literature reviews.",
            user_prompt=pm.get(
                "detect_gaps",
                research_question=question,
                knowledge_map_summary=km_summary,
            ),
        )
    except Exception as e:
        logger.error(f"Gap detection LLM call failed: {e}")
        state["gaps"] = [Gap(
            description=f"Unable to assess gaps for: {question}",
            severity=GapSeverity.CRITICAL,
        ).model_dump()]
        return state

    gaps: list[Gap] = []
    for item in gaps_json if isinstance(gaps_json, list) else []:
        if not isinstance(item, dict):
            continue
        desc = item.get("description", "")
        if not desc:
            continue

        severity_raw = item.get("severity", "important").lower()
        try:
            severity = GapSeverity(severity_raw)
        except ValueError:
            severity = GapSeverity.IMPORTANT

        gaps.append(Gap(description=desc, severity=severity))

    if not gaps:
        gaps.append(Gap(
            description=f"No specific gaps identified for: {question}. The literature appears comprehensive for the current scope.",
            severity=GapSeverity.NICE_TO_HAVE,
        ))

    logger.info(f"[detect_gaps] Fresh detection: {len(gaps)} gaps found")
    state["gaps"] = [g.model_dump() for g in gaps]
    return state
