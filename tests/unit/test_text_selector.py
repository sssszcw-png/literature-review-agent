"""Tests for smart text selector."""

from src.utils.text_selector import select_text, _split_by_headers


class TestSplitByHeaders:
    def test_no_headers_returns_empty(self):
        assert _split_by_headers("just plain text") == []

    def test_single_header(self):
        sections = _split_by_headers("## Introduction\nSome content here.")
        assert len(sections) == 1
        assert sections[0][0] == "Introduction"
        assert "Some content" in sections[0][1]

    def test_preamble_before_header(self):
        sections = _split_by_headers("Preamble text\n## Methods\nContent")
        assert len(sections) == 2
        assert sections[0][0] is None
        assert sections[0][1] == "Preamble text"
        assert sections[1][0] == "Methods"

    def test_multiple_headers(self):
        text = "## Abstract\nblah\n## Methods\ncode\n## Results\ndata"
        sections = _split_by_headers(text)
        assert len(sections) == 3
        assert [s[0] for s in sections] == ["Abstract", "Methods", "Results"]

    def test_h3_headers(self):
        sections = _split_by_headers("### Experiment Setup\nDetails here")
        assert len(sections) == 1
        assert sections[0][0] == "Experiment Setup"


class TestSelectText:
    def test_plain_text_no_headers(self):
        text = "This is a paper without any markdown headers."
        result = select_text(text, max_chars=20)
        assert result == text[:20]

    def test_skips_abstract_and_references(self):
        text = "## Abstract\nskip me\n## Methods\nkeep me\n## References\nskip too"
        result = select_text(text, max_chars=500)
        assert "keep me" in result
        assert "skip me" not in result
        assert "skip too" not in result

    def test_prioritizes_methods_over_introduction(self):
        text = "## Introduction\nintro " + "x" * 4000 + "\n## Methods\nmethod content"
        result = select_text(text, max_chars=200)
        # Methods should appear before introduction when budget is tight
        assert "method content" in result

    def test_fills_remaining_budget_with_low_priority(self):
        text = "## Methods\nshort\n## Introduction\nintro text here"
        result = select_text(text, max_chars=500)
        assert "short" in result
        assert "intro text here" in result

    def test_truncates_section_when_over_budget(self):
        text = "## Methods\n" + "A" * 300
        result = select_text(text, max_chars=100)
        # Should include header but truncate body
        assert "## Methods" in result
        assert len(result) <= 100

    def test_full_paper_realistic(self):
        text = """## Abstract
This paper proposes a novel method for image classification.

## Introduction
Deep learning has achieved great success in recent years.

## Related Work
Many previous approaches have been proposed.

## Methodology
We propose a new architecture called FastNet that reduces parameters by 50%.

## Experiments
We evaluate on ImageNet and CIFAR-100. FastNet achieves 95.3% accuracy.

## Results and Discussion
The results show significant improvement over baselines.

## Conclusion
FastNet is effective and efficient for image classification.

## References
[1] K. He et al., Deep Residual Learning, CVPR 2016.
"""
        result = select_text(text, max_chars=8000)
        assert "Methodology" in result
        assert "Experiments" in result
        assert "Results and Discussion" in result
        assert "Conclusion" in result
        assert "Abstract" not in result
        assert "References" not in result
        # Introduction and Related Work may or may not appear depending on budget
        assert "novel architecture" in result or "FastNet" in result

    def test_unknown_section_treated_as_high(self):
        text = "## Custom Analysis\ndata here\n## Introduction\nintro"
        result = select_text(text, max_chars=200)
        assert "data here" in result
