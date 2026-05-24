"""Tests for IEEE citation formatting."""

from src.report.formatter import (
    format_ieee_rules_with_index,
    validate_citations,
)


class TestFormatIeeeRules:
    def test_includes_rules(self, sample_citation_index):
        rules = format_ieee_rules_with_index(sample_citation_index)
        assert "IEEE Citation Format Rules" in rules
        assert "[1]" in rules
        assert "Attention Is All You Need" in rules

    def test_empty_index(self):
        rules = format_ieee_rules_with_index({})
        assert "Reference list:" in rules

    def test_multiple_entries(self, sample_citation_index):
        rules = format_ieee_rules_with_index(sample_citation_index)
        assert "[1]" in rules
        assert "[2]" in rules


class TestValidateCitations:
    def test_all_valid(self, sample_citation_index):
        draft = "This is a test [1] and this too [2]."
        issues = validate_citations(draft, sample_citation_index)
        assert len(issues) == 0

    def test_missing_reference(self, sample_citation_index):
        draft = "This has [1] and a missing [99] reference."
        issues = validate_citations(draft, sample_citation_index)
        assert len(issues) >= 1
        assert "99" in issues[0]

    def test_range_citations(self, sample_citation_index):
        draft = "Multiple sources [1-2] support this."
        issues = validate_citations(draft, sample_citation_index)
        assert len(issues) == 0

    def test_no_citations(self, sample_citation_index):
        draft = "No citations in this text."
        issues = validate_citations(draft, sample_citation_index)
        assert len(issues) == 0

    def test_mixed_valid_invalid(self, sample_citation_index):
        draft = "[1] is valid but [42] and [99] are not."
        issues = validate_citations(draft, sample_citation_index)
        assert len(issues) == 2
