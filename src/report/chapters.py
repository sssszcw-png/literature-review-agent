"""Async parallel chapter writing utilities."""

from __future__ import annotations

import asyncio
import logging

from src.knowledge.models import KnowledgeNode
from src.llm.client import LLMClient
from src.llm.prompts import PromptManager

logger = logging.getLogger(__name__)


async def write_chapters_parallel(
    outline: dict,
    km_nodes: list[KnowledgeNode],
    question: str,
    ieee_rules: str,
    llm: LLMClient,
    pm: PromptManager,
    max_concurrent: int = 5,
) -> dict[str, str]:
    """Write all report chapters in parallel with concurrency limiting."""

    semaphore = asyncio.Semaphore(max_concurrent)

    async def write_one(ch: dict) -> tuple[str, str]:
        title = ch.get("title", "Untitled")
        topics = ch.get("topics", [])
        evidence = collect_evidence(topics, km_nodes)

        async with semaphore:
            try:
                content = await llm.complete(
                    system_prompt="You write academic literature review chapters.",
                    user_prompt=pm.get(
                        "write_chapter",
                        chapter_title=title,
                        chapter_topics=", ".join(topics),
                        research_question=question,
                        relevant_evidence=evidence,
                        ieee_citation_rules=ieee_rules,
                    ),
                    max_tokens=3000,
                )
                return (title, content)
            except Exception as e:
                logger.error(f"Chapter '{title}' write failed: {e}")
                return (title, f"*[Error writing chapter: {e}]*")

    tasks = [asyncio.create_task(write_one(ch)) for ch in outline.get("chapters", [])]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    chapters = {}
    for r in results:
        if isinstance(r, tuple):
            chapters[r[0]] = r[1]
        elif isinstance(r, Exception):
            logger.error(f"Chapter write exception: {r}")

    return chapters


def collect_evidence(topics: list[str], nodes: list[KnowledgeNode]) -> str:
    """Collect all evidence relevant to the given topics."""
    parts = []
    for node in nodes:
        for topic in topics:
            if topic.lower() in node.topic.lower() or topic.lower() in node.claim.lower():
                evidence_items = []
                for ev in node.evidence:
                    evidence_items.append(
                        f"  - [{ev.paper.title}] ({ev.paper.venue}, {ev.paper.year}) "
                        f"Section: {ev.section}\n    Quote: \"{ev.quote}\""
                    )
                if evidence_items:
                    parts.append(
                        f"**Topic: {node.topic}** (confidence: {node.confidence:.2f})\n"
                        f"Claim: {node.claim}\n"
                        f"Evidence:\n" + "\n".join(evidence_items)
                    )
    return "\n\n".join(parts) if parts else "(No specific evidence for these topics)"
