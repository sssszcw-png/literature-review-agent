"""Tests for paper deduplication utilities."""

from src.utils.dedup import make_dedup_key, normalize_title, titles_match


class TestNormalizeTitle:
    def test_lowercase(self):
        assert normalize_title("Hello World") == "hello world"

    def test_remove_punctuation(self):
        assert normalize_title("Hello, World!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_title("Hello   World") == "hello world"

    def test_special_chars(self):
        normalized = normalize_title("Attention: Is All You Need?")
        assert ":" not in normalized
        assert "?" not in normalized


class TestTitlesMatch:
    def test_exact_match(self):
        assert titles_match("Attention Is All You Need", "Attention Is All You Need")

    def test_case_insensitive(self):
        assert titles_match("ATTENTION IS ALL YOU NEED", "attention is all you need")

    def test_high_similarity(self):
        assert titles_match(
            "Attention Is All You Need",
            "Attention Is All You Need For NLP",
            threshold=0.8,
        )

    def test_low_similarity(self):
        assert not titles_match(
            "Attention Mechanisms",
            "Convolutional Neural Networks",
            threshold=0.8,
        )

    def test_punctuation_difference(self):
        assert titles_match(
            "Attention Is All You Need",
            "Attention: Is All You Need?",
        )


class TestMakeDedupKey:
    def test_basic_key(self):
        key = make_dedup_key("Attention Is All You Need", "Vaswani", "2017")
        assert "attention is all you need" in key
        assert "vaswani" in key
        assert "2017" in key

    def test_no_optional_fields(self):
        key = make_dedup_key("Some Paper")
        assert key == "some paper"

    def test_different_keys_for_different_titles(self):
        key_a = make_dedup_key("Paper A", "Smith", "2020")
        key_b = make_dedup_key("Paper B", "Smith", "2020")
        assert key_a != key_b
