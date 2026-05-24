"""Conditional routing functions for the LangGraph state machine."""

from __future__ import annotations

import logging

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

SATURATION_THRESHOLD = 0.7
MAX_NO_IMPROVEMENT = 2


def route_after_detect_gaps(state: AgentState) -> str:
    """Decide next action after gap detection.

    Phase "broad" (Round 1 just finished):
        → plan_queries (start Round 2 deep dive)

    Phase "deep_dive" (Round 2+):
        If saturation not yet evaluated for new round:
            → evaluate_saturation
        Elif critical gaps remain, round < max, and improving:
            → plan_queries (continue deep dive)
        Else:
            → generate_outline
    """
    phase = state.get("phase", "broad")
    gaps = state.get("gaps", [])
    max_rounds = state.get("max_rounds", 5)
    current_round = state.get("current_round", 1)
    consecutive_no_improvement = state.get("consecutive_no_improvement", 0)

    if phase == "broad":
        logger.info("Routing: Round 1 complete → starting deep dive (plan_queries)")
        return "plan_queries"

    # Deep dive phase: check if saturation needs evaluation
    any_saturated = any(g.get("saturation", 0.0) > 0.0 for g in gaps)
    if not any_saturated and gaps:
        logger.info("Routing: saturation not yet evaluated → evaluate_saturation")
        return "evaluate_saturation"

    # Saturation has been evaluated — check if we should continue
    critical_unsaturated = [
        g for g in gaps
        if g.get("severity") == "critical" and g.get("saturation", 0.0) < SATURATION_THRESHOLD
    ]

    if not critical_unsaturated:
        logger.info("Routing: all critical gaps saturated → generate_outline")
        return "generate_outline"

    if current_round >= max_rounds:
        logger.info(f"Routing: max rounds ({max_rounds}) reached → generate_outline")
        return "generate_outline"

    if consecutive_no_improvement >= MAX_NO_IMPROVEMENT:
        logger.info(
            f"Routing: no improvement for {consecutive_no_improvement} rounds → generate_outline"
        )
        return "generate_outline"

    logger.info(
        f"Routing: {len(critical_unsaturated)} critical gaps unsaturated, "
        f"round {current_round}/{max_rounds} → continue deep dive"
    )
    return "plan_queries"


def route_after_generate_outline(state: AgentState) -> str:
    """Handle HITL decision after outline generation.

    user_action:
        "approve" → write_report
        "edit"    → generate_outline (re-generate with feedback)
        "abort"   → END
    """
    action = state.get("user_action", "approve")

    if action == "abort":
        logger.info("Routing: user aborted → END")
        return "__end__"
    elif action == "edit":
        logger.info("Routing: user requested edits → regenerate outline")
        return "generate_outline"
    else:
        logger.info("Routing: outline approved → write_report")
        return "write_report"
