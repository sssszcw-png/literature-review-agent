"""Paper deduplication utilities."""

import re
from difflib import SequenceMatcher


def normalize_title(title: str) -> str:
    """Normalize a paper title for comparison."""
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def titles_match(title_a: str, title_b: str, threshold: float = 0.85) -> bool:
    """Check if two paper titles likely refer to the same paper."""
    a = normalize_title(title_a)
    b = normalize_title(title_b)
    if a == b:
        return True
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


def make_dedup_key(title: str, first_author_surname: str = "", year: str = "") -> str:
    """Create a deduplication key from paper metadata."""
    normalized = normalize_title(title)
    parts = [normalized]
    if first_author_surname:
        parts.append(first_author_surname.lower())
    if year:
        parts.append(year)
    return "|".join(parts)
