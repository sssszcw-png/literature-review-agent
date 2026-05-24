"""Report outline data model and generation utilities."""

from __future__ import annotations

import json
from pathlib import Path

from src.knowledge.models import Outline, OutlineItem
from src.llm.client import LLMClient
from src.llm.prompts import PromptManager


def outline_to_markdown(outline: Outline) -> str:
    """Render an Outline as readable markdown."""
    lines = [f"# {outline.title}", ""]
    for i, ch in enumerate(outline.chapters, 1):
        lines.append(f"## {i}. {ch.title}")
        if ch.description:
            lines.append(f"*{ch.description}*")
        if ch.topics:
            lines.append(f"Topics: {', '.join(ch.topics)}")
        lines.append("")
    return "\n".join(lines)


def outline_from_dict(data: dict) -> Outline:
    """Construct an Outline from a dict (e.g. LLM response)."""
    chapters = [
        OutlineItem(
            title=ch.get("title", ""),
            topics=ch.get("topics", []),
            description=ch.get("description", ""),
        )
        for ch in data.get("chapters", [])
    ]
    return Outline(
        title=data.get("title", "Literature Review"),
        chapters=chapters,
        metadata=data.get("metadata", {}),
    )
