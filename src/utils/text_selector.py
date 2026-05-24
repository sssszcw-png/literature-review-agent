"""Smart text selection for academic papers — prioritize high-value sections."""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Sections to skip entirely (already covered by abstract or background)
SKIP_SECTIONS = {
    "abstract", "acknowledgments", "acknowledgements", "references",
    "bibliography", "appendix", "supplementary", "supplementary material",
    "supplemental", "supplemental material",
}

# Low-priority sections (background context, not novel content)
LOW_PRIORITY = {
    "introduction", "related work", "related works", "literature review",
    "background", "preliminaries", "motivation",
}

# High-priority sections (core novel content)
HIGH_PRIORITY = {
    "method", "methods", "methodology", "approach", "proposed method",
    "proposed approach", "our method", "our approach", "model",
    "architecture", "system design", "design", "implementation",
    "algorithm", "algorithms",
    "experiment", "experiments", "experimental setup", "experimental results",
    "results", "evaluation", "results and discussion", "results and analysis",
    "discussion", "analysis", "findings", "conclusion", "conclusions",
    "conclusion and future work", "summary", "limitations",
    "ablation", "ablation study", "ablation studies", "case study", "case studies",
    "comparison", "benchmark", "benchmarks",
}


def select_text(text: str, max_chars: int = 8000) -> str:
    """Select the most valuable portions of a paper, up to max_chars.

    For Markdown with ## section headers, prioritizes high-value sections
    (methods, experiments, results, discussion) over low-value ones
    (introduction, related work). Falls back to simple truncation if no
    section headers are found.

    Ordering within each priority tier is preserved from the original paper.
    """
    sections = _split_by_headers(text)

    if not sections:
        # No headers found — plain text, use as-is
        return text[:max_chars]

    if len(sections) == 1 and sections[0][0] is None:
        return text[:max_chars]

    high: list[str] = []
    low: list[str] = []
    skipped = 0

    for title, body in sections:
        if title is None:
            # Preamble before any heading — keep if short, skip if long
            if len(body) <= 500:
                high.append(body)
            else:
                low.append(body)
            continue

        normalized = title.strip().lower().rstrip(".")
        if normalized in SKIP_SECTIONS:
            skipped += 1
            continue
        elif normalized in HIGH_PRIORITY:
            high.append(f"## {title}\n{body}")
        elif normalized in LOW_PRIORITY:
            low.append(f"## {title}\n{body}")
        else:
            # Unknown sections → treat as medium: after high, before low
            high.append(f"## {title}\n{body}")

    # Build result: high first, then low to fill remaining budget
    result_parts: list[str] = []
    remaining = max_chars

    for section_text in high:
        if len(section_text) <= remaining:
            result_parts.append(section_text)
            remaining -= len(section_text)
        else:
            result_parts.append(section_text[:remaining])
            remaining = 0
            break

    # If budget remains, add low-priority sections
    if remaining > 0:
        for section_text in low:
            if len(section_text) <= remaining:
                result_parts.append(section_text)
                remaining -= len(section_text)
            else:
                result_parts.append(section_text[:remaining])
                remaining = 0
                break

    result = "\n\n".join(result_parts)
    used = len(result)
    logger.debug(
        f"Text selection: {len(sections)} sections → "
        f"{len(high)} high, {len(low)} low, {skipped} skipped, "
        f"{used}/{max_chars} chars used"
    )
    return result


def _split_by_headers(text: str) -> list[tuple[str | None, str]]:
    """Split markdown text into (section_title, section_body) pairs.

    Handles ## and ### level headers. Preamble before the first header
    is returned as (None, preamble_text).
    """
    # Match ## or ### headers at line start
    header_pattern = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)

    matches = list(header_pattern.finditer(text))
    if not matches:
        return []

    sections: list[tuple[str | None, str]] = []

    # Preamble before first header
    first_start = matches[0].start()
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.append((None, preamble))

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((title, body))

    return sections
