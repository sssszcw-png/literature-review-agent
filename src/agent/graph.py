"""LangGraph state machine construction — 7 nodes, conditional routing, checkpoints."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agent.routing import route_after_detect_gaps, route_after_generate_outline
from src.agent.state import AgentState
from src.nodes.plan_queries import plan_queries_node
from src.nodes.search import search_node
from src.nodes.read import read_node
from src.nodes.update_knowledge_map import update_knowledge_map_node
from src.nodes.detect_gaps import detect_gaps_node
from src.nodes.generate_outline import generate_outline_node
from src.nodes.write_report import write_report_node
from src.nodes.evaluate_saturation import evaluate_saturation_node

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Build and compile the Academic Research Agent state graph.

    Graph topology:

        START
          │
    plan_queries ──→ search ──→ read ──→ update_knowledge_map ──→ detect_gaps
          ↑                                                              │
          │                                          ┌───────────────────┤
          │                                          │ (gaps saturated or │
          │         (critical gaps remain)           │  max rounds)       │
          │                                          ↓                    │
          └────────────────────────────── evaluate_saturation             │
                                                         │               │
                                                         └──→ detect_gaps┤
                                                                         ↓
                                                                 generate_outline
                                                                         │
                                                              ┌──────────┤
                                                              │ HITL     │ (abort)
                                                              ↓          ↓
                                                        write_report    END
                                                              │
                                                             END
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ──
    builder.add_node("plan_queries", plan_queries_node)
    builder.add_node("search", search_node)
    builder.add_node("read", read_node)
    builder.add_node("update_knowledge_map", update_knowledge_map_node)
    builder.add_node("detect_gaps", detect_gaps_node)
    builder.add_node("evaluate_saturation", evaluate_saturation_node)
    builder.add_node("generate_outline", generate_outline_node)
    builder.add_node("write_report", write_report_node)

    # ── Edges ──
    builder.set_entry_point("plan_queries")
    builder.add_edge("plan_queries", "search")
    builder.add_edge("search", "read")
    builder.add_edge("read", "update_knowledge_map")
    builder.add_edge("update_knowledge_map", "detect_gaps")

    # Conditional: detect_gaps → plan_queries (loop) OR evaluate_saturation OR generate_outline
    builder.add_conditional_edges(
        "detect_gaps",
        route_after_detect_gaps,
        {
            "plan_queries": "plan_queries",
            "evaluate_saturation": "evaluate_saturation",
            "generate_outline": "generate_outline",
        },
    )

    # evaluate_saturation feeds back to detect_gaps to check if we advance
    builder.add_edge("evaluate_saturation", "detect_gaps")

    # Conditional: generate_outline → write_report OR generate_outline (edit) OR END (abort)
    builder.add_conditional_edges(
        "generate_outline",
        route_after_generate_outline,
        {
            "write_report": "write_report",
            "generate_outline": "generate_outline",
            "__end__": END,
        },
    )

    builder.add_edge("write_report", END)

    # ── Checkpoint ──
    # MemorySaver: LangGraph internal state tracking (required for graph execution).
    # For persistent crash recovery, CheckpointManager (src/utils/checkpoints.py)
    # saves state to disk after each significant node in the CLI app.
    checkpointer = MemorySaver()
    compiled = builder.compile(checkpointer=checkpointer)

    logger.info("Graph compiled successfully with 8 nodes")
    return compiled
