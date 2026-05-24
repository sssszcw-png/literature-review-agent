"""Async concurrency utilities."""

import asyncio
import time
from typing import Awaitable, TypeVar

T = TypeVar("T")


class RateLimiter:
    """Token bucket rate limiter for async calls."""

    def __init__(self, rate: float):
        """rate: max calls per second."""
        self.rate = rate
        self._tokens = rate
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.rate, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self.rate
                await asyncio.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


async def bounded_gather(
    *coros: Awaitable[T],
    max_concurrent: int = 5,
    return_exceptions: bool = True,
) -> list[T | Exception]:
    """Run coroutines with a concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro  # type: ignore[no-any-return]

    tasks = [asyncio.create_task(run(c)) for c in coros]
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)  # type: ignore[return-value]
