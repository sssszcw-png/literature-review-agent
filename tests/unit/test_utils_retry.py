"""Tests for async retry decorator."""

import asyncio

import pytest

from src.utils.retry import async_retry


class TestAsyncRetry:
    async def test_success_first_try(self):
        call_count = 0

        @async_retry(max_attempts=3, backoff_base=0.01)
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeed()
        assert result == "ok"
        assert call_count == 1

    async def test_retry_then_succeed(self):
        call_count = 0

        @async_retry(max_attempts=3, backoff_base=0.01)
        async def flakey():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "recovered"

        result = await flakey()
        assert result == "recovered"
        assert call_count == 3

    async def test_exhaust_retries(self):
        call_count = 0

        @async_retry(max_attempts=3, backoff_base=0.01)
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent error")

        with pytest.raises(RuntimeError, match="permanent error"):
            await always_fail()
        assert call_count == 3

    async def test_specific_exception_only(self):
        call_count = 0

        @async_retry(max_attempts=3, backoff_base=0.01, exceptions=(ValueError,))
        async def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retried")

        with pytest.raises(TypeError):
            await raise_type_error()
        assert call_count == 1  # Not retried — TypeError not in exceptions tuple

    async def test_arguments_preserved(self):
        @async_retry(max_attempts=2, backoff_base=0.01)
        async def with_args(a, b, c=3):
            return a + b + c

        result = await with_args(1, 2, c=4)
        assert result == 7
