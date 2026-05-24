"""Hardcoded constants that should NOT be user-configurable."""

# Round limits
MAX_QUERIES_PER_ROUND = 8
MIN_QUERIES_PER_ROUND = 1

# Paper ID prefixes
PAPER_ID_PREFIX_SS = "ss"
PAPER_ID_PREFIX_ARXIV = "arxiv"

# Evidence types
SUPPORTED_EVIDENCE_TYPES = frozenset({"original_claim", "supporting", "contradicting"})

# Gap severity levels
SUPPORTED_SEVERITY_LEVELS = frozenset({"critical", "important", "nice_to_have"})

# IEEE citation format rules (injected into LLM prompts)
IEEE_CITATION_RULES = """
IEEE Citation Format Rules:
- Inline citations use numbered brackets: [1], [2-4]
- Every claim or fact from a paper MUST include a citation
- Reference list in "## References" section at the end of the report
- Each entry format: [N] Author(s), "Title," *Venue*, Year. [URL](url)
- Number references in order of first appearance
- For multiple citations at the same point, use [1], [2], not [1, 2]
"""

# Phase markers
PHASE_BROAD = "broad"
PHASE_DEEP_DIVE = "deep_dive"
PHASE_REPORT = "report"

# Report generation
CONSISTENCY_CHECK_MAX_CHARS = 12000
