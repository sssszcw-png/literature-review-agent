"""CLI application: orchestrates the graph run with Rich display."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.graph import build_graph
from src.cli.progress import ResearchProgress
from src.config.settings import Settings
from src.utils.checkpoints import CheckpointManager
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class ResearchCLI:
    """Orchestrates the full research pipeline with progress display."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.progress = ResearchProgress()
        self.final_report_zh: str = ""

    async def run(
        self,
        question: str,
        max_rounds: int = 5,
        output_dir: str = "reports",
        resume_thread_id: Optional[str] = None,
        zh: bool = False,
    ) -> str:
        """Execute the full research pipeline and return the final report."""

        graph = build_graph()

        initial_state: AgentState = {
            "research_question": question,
            "max_rounds": max_rounds,
            "output_dir": output_dir,
            "current_round": 0,
            "phase": "broad",
            "round_history": [],
            "search_queries": [],
            "paper_index": {},
            "search_results": [],
            "papers_read": {},
            "read_failures": {},
            "knowledge_nodes": [],
            "gaps": [],
            "outline": None,
            "outline_feedback": "",
            "outline_approved": False,
            "chapters": {},
            "final_report": "",
            "messages": [],
            "errors": [],
            "iteration_count": 0,
            "last_saturation_scores": {},
            "consecutive_no_improvement": 0,
            "checkpoint_metadata": {},
            "user_action": "approve",
            "output_zh": zh,
            "final_report_zh": "",
        }

        thread_id = resume_thread_id or str(int(time.time()))
        config = {"configurable": {"thread_id": thread_id}}

        # Resume from checkpoint if requested
        if resume_thread_id:
            ckmgr = CheckpointManager(self.settings.checkpoint_dir)
            saved = ckmgr.load(resume_thread_id)
            if saved:
                logger.info(f"Resuming from checkpoint: {resume_thread_id}")
                initial_state.update(saved)  # type: ignore[typeddict-item]
        start_time = time.time()

        self.progress.start(question)

        ckmgr = CheckpointManager(self.settings.checkpoint_dir) if self.settings.checkpoint_enabled else None

        node_state = None
        try:
            async for event in graph.astream(initial_state, config=config):
                for node_name, node_state in event.items():
                    self.progress.update(node_name, node_state)

                    # Save checkpoint after each significant node
                    if ckmgr and node_name not in ("plan_queries",):
                        ckmgr.save(node_state, thread_id, node_name)

                    # Update round tracking
                    if node_name == "plan_queries":
                        current = node_state.get("current_round", 0)
                        if current == 0:
                            node_state["current_round"] = 1
                            node_state["phase"] = "broad"
        except KeyboardInterrupt:
            self.progress.stop("Interrupted by user")
            raise
        except Exception as e:
            self.progress.stop(f"Error: {e}")
            raise

        elapsed = time.time() - start_time
        self.progress.stop(f"Completed in {elapsed:.1f}s")

        # Get final state from the last event
        final_report = node_state.get("final_report", "") if node_state is not None else ""
        self.final_report_zh = node_state.get("final_report_zh", "") if node_state is not None else ""

        return final_report
