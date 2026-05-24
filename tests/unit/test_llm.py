"""Tests for LLMClient and PromptManager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.client import LLMClient
from src.llm.prompts import PromptManager
from src.config.settings import Settings


class TestPromptManager:
    @pytest.fixture
    def prompt_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "test_prompt.txt").write_text(
                "Question: {question}\nCount: {num}", encoding="utf-8"
            )
            (d / "another.txt").write_text("Hello {name}", encoding="utf-8")
            yield d

    def test_loads_all_templates(self, prompt_dir):
        pm = PromptManager(prompt_dir)
        assert "test_prompt" in pm._templates
        assert "another" in pm._templates

    def test_get_renders_template(self, prompt_dir):
        pm = PromptManager(prompt_dir)
        result = pm.get("test_prompt", question="What is X?", num=5)
        assert "What is X?" in result
        assert "5" in result

    def test_get_unknown_prompt_raises(self, prompt_dir):
        pm = PromptManager(prompt_dir)
        with pytest.raises(KeyError, match="Unknown prompt"):
            pm.get("nonexistent")

    def test_get_missing_variable_raises(self, prompt_dir):
        pm = PromptManager(prompt_dir)
        with pytest.raises(KeyError):
            pm.get("another")  # missing name=

    def test_list_prompts(self, prompt_dir):
        pm = PromptManager(prompt_dir)
        prompts = pm.list_prompts()
        assert "another" in prompts
        assert "test_prompt" in prompts

    def test_raises_on_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            PromptManager(Path("/nonexistent/prompts"))


class TestLLMClient:
    @pytest.fixture
    def settings(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
        return Settings()

    @pytest.fixture
    def client(self, settings):
        return LLMClient(settings)

    async def test_complete(self, client):
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello, world!"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await client.complete(
                system_prompt="You are helpful.",
                user_prompt="Say hello",
            )
            assert result == "Hello, world!"

    async def test_complete_json(self, client):
        mock_choice = MagicMock()
        mock_choice.message.content = '{"key": "value", "list": [1, 2, 3]}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await client.complete_json(
                system_prompt="Extract JSON.",
                user_prompt="Return data",
            )
            assert result == {"key": "value", "list": [1, 2, 3]}

    async def test_complete_json_with_markdown_fences(self, client):
        mock_choice = MagicMock()
        mock_choice.message.content = '```json\n{"key": "value"}\n```'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await client.complete_json(
                system_prompt="Extract JSON.",
                user_prompt="Return data",
            )
            assert result == {"key": "value"}

    def test_extract_json_plain(self, client):
        result = client._extract_json('{"a": 1}')
        assert result == {"a": 1}

    def test_extract_json_with_fences(self, client):
        result = client._extract_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_extract_json_with_surrounding_text(self, client):
        result = client._extract_json('Here is the result: {"a": 1}. End.')
        assert result == {"a": 1}

    def test_extract_json_array(self, client):
        result = client._extract_json('[{"a": 1}, {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_extract_json_trailing_comma(self, client):
        result = client._extract_json('{"a": 1, "b": 2,}')
        assert result == {"a": 1, "b": 2}

    def test_extract_json_latex_escapes(self, client):
        result = client._extract_json(r'{"formula": "\\cdot \\frac{1}{2}"}')
        assert result == {"formula": r"\cdot \frac{1}{2}"}

    def test_extract_json_no_json(self, client):
        with pytest.raises(json.JSONDecodeError):
            client._extract_json("no json here at all")

    async def test_retry_on_failure(self, client):
        mock_choice = MagicMock()
        mock_choice.message.content = "success"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [
                Exception("temporary network error"),
                Exception("another error"),
                mock_response,
            ]

            # The retry decorator uses settings from self, which override the decorator params
            # We need to adjust retry params for fast test
            client.retry_max = 3
            client.retry_backoff = 0.01
            client.retry_max_delay = 0.1

            result = await client.complete(
                system_prompt="test",
                user_prompt="test",
            )
            assert result == "success"
            assert mock_create.call_count == 3
