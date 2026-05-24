"""Node: write_report — parallel chapter writing, stitching, IEEE formatting, output."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from src.agent.state import AgentState
from src.config.constants import CONSISTENCY_CHECK_MAX_CHARS
from src.config.settings import get_settings
from src.knowledge.models import KnowledgeNode
from src.llm.client import get_llm_client
from src.llm.prompts import get_prompt_manager
from src.report.stitch import build_citation_index, stitch_chapters
from src.report.formatter import format_ieee_rules_with_index
from src.report.chapters import collect_evidence

logger = logging.getLogger(__name__)


async def write_report_node(state: AgentState) -> AgentState:
    """Write report chapters in parallel, stitch, apply consistency check, format, and save.

    Reads: outline, knowledge_nodes, papers_read, paper_index, research_question, output_dir
    Writes: chapters, final_report
    """
    outline = state.get("outline", {})
    km_data = state.get("knowledge_nodes", [])
    paper_index = state.get("paper_index", {})
    question = state.get("research_question", "")
    output_dir = state.get("output_dir", "reports")

    if not outline:
        logger.error("No outline available for report writing")
        return state

    settings = get_settings()
    llm = get_llm_client()
    pm = get_prompt_manager()

    km_nodes = [KnowledgeNode(**n) if isinstance(n, dict) else n for n in km_data]

    chapters_outline = outline.get("chapters", [])

    # Build citation index: assign a number to each unique paper
    citation_index = build_citation_index(km_nodes)
    ieee_rules = format_ieee_rules_with_index(citation_index)

    # Parallel chapter writing
    chapter_tasks = []
    for ch in chapters_outline:
        title = ch.get("title", "Untitled")
        topics = ch.get("topics", [])
        evidence_str = collect_evidence(topics, km_nodes)

        task = _write_chapter(
            title=title,
            topics=", ".join(topics),
            evidence=evidence_str,
            question=question,
            ieee_rules=ieee_rules,
            llm=llm,
            pm=pm,
        )
        chapter_tasks.append((title, task))

    # Run all chapter writes concurrently
    chapters = {}
    for title, task in chapter_tasks:
        try:
            content = await task
            chapters[title] = content
        except Exception as e:
            logger.error(f"Chapter '{title}' write failed: {e}")
            chapters[title] = f"*[Error writing chapter: {e}]*"

    # Stitch chapters
    draft = stitch_chapters(outline, chapters, citation_index)

    # Consistency check
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
            draft = check_response["fixed_report"]
    except Exception as e:
        logger.warning(f"Consistency check failed: {e}")

    final = draft

    # Write to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_name = _slugify(question)[:80]
    report_path = output_path / f"{safe_name}.md"
    report_path.write_text(final, encoding="utf-8")

    # Append errors appendix if any
    errors = state.get("errors", [])
    if errors:
        error_appendix = "\n\n## Appendix: Errors and Degradations\n\n"
        for err in errors:
            error_appendix += (
                f"- **{err.get('node', 'unknown')}**: {err.get('message', err.get('error', ''))}\n"
            )
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(error_appendix)

    state["chapters"] = chapters
    state["final_report"] = final

    # Translate to Chinese if requested
    output_zh = state.get("output_zh", False)
    if output_zh and final.strip():
        try:
            zh_report = await llm.complete(
                system_prompt="You are a professional academic translator specializing in computer science and AI literature.",
                user_prompt=pm.get("translate_to_chinese", report=final),
                max_tokens=24576,
            )
            zh_path = output_path / f"{safe_name}_zh.md"
            zh_path.write_text(zh_report, encoding="utf-8")
            state["final_report_zh"] = zh_report
            logger.info(f"[write_report] Chinese translation saved to: {zh_path}")
        except Exception as e:
            logger.warning(f"Chinese translation failed: {e}")
            state["final_report_zh"] = ""
    else:
        state["final_report_zh"] = ""

    logger.info(f"[write_report] Report saved to: {report_path}")
    return state


async def _write_chapter(
    title: str,
    topics: str,
    evidence: str,
    question: str,
    ieee_rules: str,
    llm: LLMClient,
    pm: PromptManager,
) -> str:
    """Write a single chapter asynchronously."""
    return await llm.complete(
        system_prompt="You write academic literature review chapters.",
        user_prompt=pm.get(
            "write_chapter",
            chapter_title=title,
            chapter_topics=topics,
            research_question=question,
            relevant_evidence=evidence,
            ieee_citation_rules=ieee_rules,
        ),
        max_tokens=3000,
    )


def _slugify(text: str) -> str:
    """Create a filename-safe slug from text."""
    import re
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:100]
