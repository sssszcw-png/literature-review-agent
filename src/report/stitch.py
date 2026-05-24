"""Chapter stitching and cross-chapter consistency checking."""

from __future__ import annotations

import json
import logging

from src.knowledge.models import KnowledgeNode
from src.llm.client import LLMClient
from src.llm.prompts import PromptManager
from src.config.constants import CONSISTENCY_CHECK_MAX_CHARS
from src.report.formatter import format_reference_entry

logger = logging.getLogger(__name__)


def stitch_chapters(
    outline: dict,
    chapters: dict[str, str],
    citation_index: dict[int, dict],
) -> str:
    """Stitch chapters together into a complete draft with reference list."""

    title = outline.get("title", "Literature Review")
    lines = [f"# {title}", ""]

    for ch in outline.get("chapters", []):
        ch_title = ch.get("title", "Untitled")
        content = chapters.get(ch_title, "*[Chapter not generated]*")
        lines.append(f"## {ch_title}")
        lines.append("")
        lines.append(content)
        lines.append("")

    # Reference list
    lines.append("## References")
    lines.append("")
    for num in sorted(citation_index.keys()):
        paper = citation_index[num]
        lines.append(f"[{num}] {format_reference_entry(paper)}")

    return "\n".join(lines)


def build_citation_index(nodes: list[KnowledgeNode]) -> dict[int, dict]:
    """Build a numbered citation index [1], [2], ... from all evidence."""

    seen = {}
    idx = 1
    index = {}
    for node in nodes:
        for ev in node.evidence:
            if ev.paper.paper_id not in seen:
                seen[ev.paper.paper_id] = idx
                index[idx] = {
                    "paper_id": ev.paper.paper_id,
                    "title": ev.paper.title,
                    "authors": ev.paper.authors,
                    "year": ev.paper.year,
                    "venue": ev.paper.venue,
                    "url": ev.paper.url,
                }
                idx += 1
    return index


async def check_consistency(
    draft: str,
    citation_index: dict[int, dict],
    llm: LLMClient,
    pm: PromptManager,
) -> str:
    """Run cross-chapter consistency check and return fixed draft."""

    try:
        check_response = await llm.complete_json(
            system_prompt="You perform editorial consistency checks.",
            user_prompt=pm.get(
                "consistency_check",
                full_report=draft[:CONSISTENCY_CHECK_MAX_CHARS],
                citation_index=json.dumps(citation_index, indent=2, ensure_ascii=False),
            ),
        )
        if check_response.get("has_significant_problems") and check_response.get("fixed_report"):
            return check_response["fixed_report"]
    except Exception as e:
        logger.warning(f"Consistency check failed: {e}")

    return draft
