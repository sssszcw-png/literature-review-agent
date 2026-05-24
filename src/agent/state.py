"""AgentState TypedDict — the central state flowing through the LangGraph state machine."""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class RoundSummary(TypedDict, total=False):
    round: int
    phase: str
    queries_count: int
    papers_found: int
    papers_read: int
    nodes_added: int
    gaps_detected: int
    saturation_scores: dict[str, float]


class AgentState(TypedDict, total=False):
    # ── Input ──
    research_question: str
    max_rounds: int
    output_dir: str

    # ── Round Tracking ──
    current_round: int
    phase: str  # "broad", "deep_dive", "report"
    round_history: list[RoundSummary]

    # ── Query Planning ──
    search_queries: list[str]

    # ── Search ──
    paper_index: dict[str, dict]  # paper_id -> PaperMeta (serialized as dict)
    search_results: list[str]  # paper IDs from latest search (not yet read)

    # ── Reading ──
    papers_read: dict[str, str]  # paper_id -> full text or abstract markdown
    read_failures: dict[str, str]  # paper_id -> failure reason

    # ── Knowledge ──
    knowledge_nodes: list[dict]  # list of KnowledgeNode (serialized)
    gaps: list[dict]  # list of Gap (serialized)

    # ── Report ──
    outline: Optional[dict]  # Outline serialized, None until generated
    outline_feedback: str  # user feedback during HITL
    outline_approved: bool
    chapters: dict[str, str]  # chapter_title -> markdown content
    final_report: str
    output_zh: bool  # whether to produce Chinese translation
    final_report_zh: str  # Chinese translated report text

    # ── Control ──
    messages: Annotated[list, add_messages]
    errors: list[dict]
    iteration_count: int
    last_saturation_scores: dict[str, float]
    consecutive_no_improvement: int
    checkpoint_metadata: dict[str, Any]

    # ── UI / Control flags ──
    user_action: str  # "approve" | "edit" | "abort" set by HITL node
