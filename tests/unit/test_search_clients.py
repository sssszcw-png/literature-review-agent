"""Tests for Semantic Scholar and arXiv API clients."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.settings import Settings
from src.search.models import SearchResult
from src.search.semantic_scholar import SemanticScholarClient
from src.search.arxiv import ArxivClient


class TestSemanticScholarClient:
    @pytest.fixture
    def settings(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        return Settings()

    @pytest.fixture
    def client(self, settings):
        return SemanticScholarClient(settings)

    @pytest.fixture
    def mock_response(self):
        return {
            "data": [
                {
                    "paperId": "abc123",
                    "externalIds": {"DOI": "10.1234/test", "ArXiv": "1706.03762"},
                    "title": "Attention Is All You Need",
                    "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
                    "year": 2017,
                    "venue": "NeurIPS",
                    "citationCount": 100000,
                    "abstract": "The dominant sequence transduction models...",
                    "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762.pdf"},
                },
                {
                    "paperId": "def456",
                    "externalIds": {},
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors": [{"name": "Jacob Devlin"}],
                    "year": 2019,
                    "venue": "NAACL",
                    "citationCount": 50000,
                    "abstract": "We introduce a new language representation model...",
                    "openAccessPdf": None,
                },
            ]
        }

    async def test_search_returns_results(self, client, mock_response):
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value = mock_resp

            results = await client.search("attention mechanism")

            assert len(results) == 2
            assert results[0].paper_id == "ss:abc123"
            assert results[0].title == "Attention Is All You Need"
            assert results[0].source == "semantic_scholar"
            assert results[0].doi == "10.1234/test"
            assert results[0].pdf_url == "https://arxiv.org/pdf/1706.03762.pdf"
            assert results[0].citation_count == 100000

    async def test_search_empty_response(self, client):
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value={"data": []})
            mock_get.return_value.__aenter__.return_value = mock_resp

            results = await client.search("nonexistent topic")
            assert len(results) == 0

    async def test_search_missing_paper_id_skipped(self, client):
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value={
                "data": [
                    {"paperId": "", "title": "No ID paper"},
                    {"paperId": "valid1", "title": "Valid paper"},
                ]
            })
            mock_get.return_value.__aenter__.return_value = mock_resp

            results = await client.search("test")
            assert len(results) == 1
            assert results[0].paper_id == "ss:valid1"

    async def test_search_timeout_returns_empty(self, client):
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.side_effect = asyncio.TimeoutError

            results = await client.search("test")
            assert len(results) == 0

    async def test_get_paper_by_id(self, client):
        mock_data = {
            "paperId": "abc123",
            "externalIds": {"DOI": "10.1234/test"},
            "title": "Test Paper",
            "authors": [{"name": "Author One"}],
            "year": 2023,
            "venue": "Test Venue",
            "citationCount": 42,
            "abstract": "Test abstract",
            "openAccessPdf": None,
        }
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_data)
            mock_get.return_value.__aenter__.return_value = mock_resp

            result = await client.get_paper_by_id("ss:abc123")
            assert result is not None
            assert result.paper_id == "ss:abc123"
            assert result.citation_count == 42

    async def test_get_paper_by_id_not_found(self, client):
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value={})
            mock_get.return_value.__aenter__.return_value = mock_resp

            result = await client.get_paper_by_id("ss:nonexistent")
            assert result is None


ARXIV_XML_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Attention Is All You Need</title>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <summary>The dominant sequence transduction models...</summary>
    <published>2017-06-12T00:00:00Z</published>
    <link rel="alternate" href="https://arxiv.org/abs/1706.03762"/>
    <link title="pdf" rel="related" href="https://arxiv.org/pdf/1706.03762.pdf"/>
  </entry>
  <entry>
    <title>BERT: Pre-training</title>
    <author><name>Jacob Devlin</name></author>
    <summary>We introduce BERT...</summary>
    <published>2019-05-24T00:00:00Z</published>
    <link rel="alternate" href="https://arxiv.org/abs/1810.04805"/>
    <link title="pdf" rel="related" href="https://arxiv.org/pdf/1810.04805.pdf"/>
  </entry>
</feed>"""


class TestArxivClient:
    @pytest.fixture
    def settings(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        return Settings()

    @pytest.fixture
    def client(self, settings):
        return ArxivClient(settings)

    async def test_search_returns_results(self, client):
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.text = AsyncMock(return_value=ARXIV_XML_RESPONSE)
            mock_get.return_value.__aenter__.return_value = mock_resp

            results = await client.search("attention mechanism")

            assert len(results) == 2
            assert results[0].paper_id == "arxiv:1706.03762"
            assert results[0].title == "Attention Is All You Need"
            assert results[0].authors == ["Ashish Vaswani", "Noam Shazeer"]
            assert results[0].year == 2017
            assert results[0].source == "arxiv"
            assert results[0].pdf_url == "https://arxiv.org/pdf/1706.03762.pdf"

    async def test_search_timeout_returns_empty(self, client):
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.side_effect = asyncio.TimeoutError

            results = await client.search("test")
            assert len(results) == 0

    async def test_search_bad_xml_returns_empty(self, client):
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.text = AsyncMock(return_value="not valid xml>")
            mock_get.return_value.__aenter__.return_value = mock_resp

            results = await client.search("test")
            assert len(results) == 0

    async def test_search_empty_feed(self, client):
        empty_xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.text = AsyncMock(return_value=empty_xml)
            mock_get.return_value.__aenter__.return_value = mock_resp

            results = await client.search("nonexistent")
            assert len(results) == 0

    async def test_search_entry_without_arxiv_id(self, client):
        """Paper without an arxiv ID in URL uses the full URL as fallback ID."""
        xml_no_id = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>Some Paper</title>
            <author><name>Author</name></author>
            <summary>Abstract text</summary>
            <published>2020-01-01T00:00:00Z</published>
            <link rel="alternate" href="https://example.com/paper"/>
          </entry>
        </feed>"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.text = AsyncMock(return_value=xml_no_id)
            mock_get.return_value.__aenter__.return_value = mock_resp

            results = await client.search("test")
            assert len(results) == 1
            # URL doesn't contain /abs/, so the ID is derived from the full URL
            assert results[0].title == "Some Paper"
            assert results[0].source == "arxiv"
