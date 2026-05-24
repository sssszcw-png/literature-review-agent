"""IEEE citation formatting for final report output."""

from __future__ import annotations

import re

from src.config.constants import IEEE_CITATION_RULES


def format_reference_entry(paper: dict) -> str:
    """Format a single reference entry in IEEE style.

    Returns: "[N] Authors, \"Title,\" *Venue*, Year. [URL](url)"
    The caller is responsible for prepending the citation number.
    """
    authors = ", ".join(paper["authors"][:3])
    if len(paper["authors"]) > 3:
        authors += " et al."
    year = paper.get("year", "n.d.")
    venue = paper.get("venue", "")
    venue_str = f", *{venue}*" if venue else ""
    url = paper.get("url", "")
    url_str = f" [{url}]({url})" if url else ""
    return f'{authors}, "{paper["title"]},"{venue_str} {year}.{url_str}'


def format_ieee_rules_with_index(citation_index: dict[int, dict]) -> str:
    """Format IEEE citation rules with the actual reference list included."""

    rules = IEEE_CITATION_RULES
    rules += "\n\nReference list:\n"
    for num in sorted(citation_index.keys()):
        paper = citation_index[num]
        rules += f"[{num}] {format_reference_entry(paper)}\n"
    return rules


def validate_citations(draft: str, citation_index: dict[int, dict]) -> list[str]:
    """Check that all inline citations [N] have a matching reference. Returns issues."""
    issues = []
    inline_citations = set()
    for match in re.finditer(r"\[(\d+(?:-\d+)?)\]", draft):
        numbers = _parse_citation_range(match.group(1))
        inline_citations.update(numbers)

    valid_numbers = set(citation_index.keys())

    for num in inline_citations:
        if num not in valid_numbers:
            issues.append(f"Citation [{num}] has no matching reference entry")

    return issues


def _parse_citation_range(text: str) -> list[int]:
    """Parse citation ranges like '2-4' into [2, 3, 4]."""
    if "-" in text:
        parts = text.split("-")
        try:
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end + 1))
        except (ValueError, IndexError):
            return []
    try:
        return [int(text)]
    except ValueError:
        return []
