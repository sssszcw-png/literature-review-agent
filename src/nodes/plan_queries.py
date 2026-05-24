"""Node: plan_queries — decompose research question into search queries."""

from __future__ import annotations

import json
import logging

from src.agent.state import AgentState
from src.config.constants import PHASE_BROAD, MAX_QUERIES_PER_ROUND
from src.config.settings import get_settings
from src.llm.client import get_llm_client
from src.llm.prompts import get_prompt_manager

logger = logging.getLogger(__name__)


async def plan_queries_node(state: AgentState) -> AgentState:
    """Generate search queries based on the current research phase.

    Round 1 (broad): decompose the research question broadly.
    Round 2+ (deep_dive): generate targeted queries for critical gaps.

    Also manages round transitions: increments current_round and sets phase.
    """
    current_round = state.get("current_round", 0)
    phase = state.get("phase", PHASE_BROAD)
    research_question = state.get("research_question", "")
    gaps = state.get("gaps", [])

    # Round transition management
    if current_round == 0:
        # First invocation: start Round 1 (broad)
        current_round = 1
        phase = PHASE_BROAD
    elif phase == PHASE_BROAD:
        # Moving from Round 1 to Round 2
        current_round += 1
        phase = "deep_dive"
    else:
        # Already in deep dive, increment round
        current_round += 1

    state["current_round"] = current_round
    state["phase"] = phase
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    logger.info(f"[plan_queries] round={current_round}, phase={phase}")

    settings = get_settings()
    llm = get_llm_client()
    pm = get_prompt_manager()

    if phase == PHASE_BROAD:
        queries = await _broad_queries(research_question, llm, pm)
    else:
        queries = await _targeted_queries(research_question, gaps, llm, pm)

    state["search_queries"] = queries
    return state


async def _broad_queries(
    question: str, llm: LLMClient, pm: PromptManager
) -> list[str]:
    """Round 1: broad decomposition of the research question."""
    try:
        response = await llm.complete(
            system_prompt="You decompose research questions for academic search.",
            user_prompt=pm.get(
                "plan_queries_broad",
                research_question=question,
                num_queries=MAX_QUERIES_PER_ROUND,
            ),
        )
        queries = _parse_numbered_list(response)
        if not queries:
            queries = [question]
        return queries[:MAX_QUERIES_PER_ROUND]
    except Exception as e:
        logger.error(f"Query planning failed: {e}")
        return [question]


async def _targeted_queries(
    question: str, gaps: list[dict], llm: LLMClient, pm: PromptManager
) -> list[str]:
    """Round 2+: targeted queries for filling critical gaps."""
    critical_gaps = [
        g for g in gaps
        if g.get("severity") == "critical" and g.get("saturation", 0.0) < 0.7
    ]
    if not critical_gaps:
        critical_gaps = [g for g in gaps if g.get("severity") == "important"]

    gaps_context = json.dumps(critical_gaps, indent=2)

    try:
        num_queries = min(len(critical_gaps) * 2, MAX_QUERIES_PER_ROUND)
        response = await llm.complete(
            system_prompt="You generate targeted academic search queries.",
            user_prompt=pm.get(
                "plan_queries_targeted",
                research_question=question,
                gaps_context=gaps_context,
                num_queries=num_queries,
            ),
        )
        queries = _parse_numbered_list(response)
        return queries[:MAX_QUERIES_PER_ROUND]
    except Exception as e:
        logger.error(f"Targeted query planning failed: {e}")
        return [g.get("description", question) for g in critical_gaps[:3]]


def _parse_numbered_list(text: str) -> list[str]:
    """Parse a numbered list response into individual queries.

    Handles formats like:
      "1. query text"
      "1) query text"
      "1 - query text"
      "1 query text"
      "- query text" (bullet)
    """
    import re

    lines = text.strip().split("\n")
    queries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading number/bullet prefix: optional digits followed by [.)\s-]+
        q = re.sub(r"^(\d+[.)\s-]+|\s*[-*]\s*)", "", line).strip()
        if q:
            queries.append(q)
    return queries
