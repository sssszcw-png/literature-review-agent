"""Checkpoint save/load/resume utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.agent.state import AgentState

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint persistence for crash recovery."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: AgentState, thread_id: str, node_name: str) -> Path:
        """Save the current agent state to disk."""

        current_round = state.get("current_round", 0)
        filename = f"{thread_id}_round{current_round}_after_{node_name}.json"
        path = self.checkpoint_dir / filename

        # Convert state to JSON-serializable form
        serializable = dict(state)
        serializable.pop("messages", None)  # Messages aren't JSON-serializable

        path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load(self, thread_id: str) -> Optional[AgentState]:
        """Load the most recent checkpoint for a thread."""

        pattern = f"{thread_id}_*.json"
        files = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            logger.warning(f"No checkpoints found for thread: {thread_id}")
            return None

        path = files[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info(f"Checkpoint loaded: {path}")
        return AgentState(**data)  # type: ignore[typeddict-item]

    def list_threads(self) -> list[str]:
        """List all thread IDs with saved checkpoints."""

        threads = set()
        for f in self.checkpoint_dir.glob("*.json"):
            thread_id = f.stem.split("_round")[0]
            threads.add(thread_id)
        return sorted(threads)

    def cleanup(self, thread_id: str) -> int:
        """Remove all checkpoints for a thread. Returns count deleted."""

        count = 0
        for f in self.checkpoint_dir.glob(f"{thread_id}_*.json"):
            f.unlink()
            count += 1
        return count
