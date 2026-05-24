"""DeepSeek API client wrapper using OpenAI-compatible SDK."""

import json
import logging
from functools import lru_cache
from typing import Optional

from openai import AsyncOpenAI

from src.config.settings import Settings, get_settings
from src.utils.retry import async_retry

logger = logging.getLogger(__name__)


class LLMClient:
    """Async wrapper around DeepSeek API via OpenAI-compatible SDK."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_request_timeout
        self.retry_max = settings.retry_max_attempts
        self.retry_backoff = settings.retry_backoff_base
        self.retry_max_delay = settings.retry_max_delay

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        result = await self._call_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return result.choices[0].message.content or ""

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Send a chat completion and parse the response as JSON."""
        full_user_prompt = f"{user_prompt}\n\nRespond with valid JSON only, no markdown fences."
        text = await self.complete(
            system_prompt=system_prompt,
            user_prompt=full_user_prompt,
            temperature=temperature or 0.0,
            max_tokens=max_tokens,
        )
        return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Robust JSON extraction from LLM output."""
        import re

        cleaned = text.strip()
        # Remove markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Try to find JSON object boundaries
        first_brace = cleaned.find("{")
        first_bracket = cleaned.find("[")
        if first_brace == -1 and first_bracket == -1:
            raise json.JSONDecodeError("No JSON object or array found", cleaned, 0)

        start = first_brace if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket) else first_bracket
        end = cleaned.rfind("}" if cleaned[start] == "{" else "]")
        if end == -1:
            end = len(cleaned)

        json_str = cleaned[start:end + 1]

        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        # Try parsing; if it fails due to invalid escapes (e.g. LaTeX \cdot),
        # fix by doubling backslashes that aren't valid JSON escapes
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            json_str = LLMClient._fix_json_escapes(json_str)
            return json.loads(json_str)

    @staticmethod
    def _fix_json_escapes(json_str: str) -> str:
        """Fix invalid escape sequences in JSON (e.g. LaTeX like \\cdot).

        Valid JSON escapes: \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX
        Any other backslash-sequence is invalid in JSON. We escape the backslash.
        """
        import re

        # Match a backslash followed by a single character that is NOT
        # a valid JSON escape start: " \ / b f n r t u
        # For \u we also need to ensure it's followed by 4 hex digits
        def _fix_escape(m: re.Match) -> str:
            char = m.group(1)
            if char in '\"\\/bfnrt':
                return m.group(0)  # valid, keep
            if char == 'u':
                # Check if followed by 4 hex digits
                rest = json_str[m.end()-1:m.end()+4]
                if re.match(r'^[0-9a-fA-F]{4}', rest):
                    return m.group(0)  # valid \uXXXX
                return '\\\\' + char  # invalid \u, escape it
            return '\\\\' + char  # escape the backslash

        return re.sub(r'\\(.)', _fix_escape, json_str)

    @async_retry(max_attempts=3, backoff_base=2.0, max_delay=60.0)
    async def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ):
        return await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )

    async def stream_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Stream a completion token by token. Yields content deltas."""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            timeout=self.timeout,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Return a cached LLMClient singleton."""
    return LLMClient(get_settings())
