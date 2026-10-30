"""Runs many generation requests against a backend with caching, retries and concurrency."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from llm_finetune_evaluator.backends.base import Backend, Generation, GenerationRequest
from llm_finetune_evaluator.backends.cache import GenerationCache, cache_key
from llm_finetune_evaluator.config import RetryConfig
from llm_finetune_evaluator.errors import BackendError, RetryableBackendError

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int], None]
"""Called with the number of requests that were just completed."""


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter for transient backend errors.

    The delay before retry ``k`` (1-based) is
    ``min(max_delay_s, initial_delay_s * multiplier ** (k - 1))``, scaled by a
    random factor in ``[1 - jitter, 1 + jitter]`` so that concurrent requests do
    not retry in lockstep. A ``Retry-After`` delay sent by the server is never
    shortened.
    """

    max_attempts: int = 5
    initial_delay_s: float = 1.0
    max_delay_s: float = 60.0
    multiplier: float = 2.0
    jitter: float = 0.2

    @classmethod
    def from_config(cls, config: RetryConfig) -> RetryPolicy:
        """Build a policy from the ``retry`` section of the config."""
        return cls(
            max_attempts=config.max_attempts,
            initial_delay_s=config.initial_delay_s,
            max_delay_s=config.max_delay_s,
            multiplier=config.multiplier,
        )

    def delay(self, retry: int, retry_after: float | None, rng: random.Random) -> float:
        """Return the delay in seconds before retry number ``retry``."""
        base = min(self.max_delay_s, self.initial_delay_s * self.multiplier ** (retry - 1))
        delay = base * (1.0 + self.jitter * (2.0 * rng.random() - 1.0))
        if retry_after is not None:
            delay = max(delay, retry_after)
        return max(0.0, delay)


@dataclass
class EngineStats:
    """Counters for one engine.

    Attributes:
        requested: Requests received.
        cached: Requests answered from the cache.
        generated: Requests sent to the backend and answered.
        retries: Retries after transient errors.
    """

    requested: int = 0
    cached: int = 0
    generated: int = 0
    retries: int = 0


class GenerationEngine:
    """Executes generation requests for one backend.

    Requests already in the cache are answered without calling the backend.
    The rest are grouped into batches of ``backend.max_batch_size`` and sent
    with at most ``backend.max_concurrency`` batches in flight. Transient
    errors are retried with exponential backoff. Each completed batch is
    written to the cache immediately, so an interrupted run resumes where it
    stopped.

    Args:
        backend: The model to call.
        cache: Generation cache, or ``None`` to disable caching.
        retry: Retry policy for transient errors.
        sleep: Coroutine used to wait between retries (replaceable in tests).
        seed: Seed for the retry jitter.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        cache: GenerationCache | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        seed: int = 0,
    ) -> None:
        self.backend = backend
        self.cache = cache if backend.cacheable else None
        self.retry = retry or RetryPolicy()
        self.stats = EngineStats()
        self._sleep = sleep
        self._rng = random.Random(seed)

    def _key(self, request: GenerationRequest) -> str:
        return cache_key(
            self.backend.identity(), self.backend.request_identity(request), request.payload()
        )

    async def generate(
        self,
        requests: Sequence[GenerationRequest],
        progress: ProgressCallback | None = None,
    ) -> list[Generation]:
        """Return one generation per request, in the order of ``requests``.

        Raises:
            BackendError: If a request fails permanently or after all retries.
                Generations completed before the failure stay in the cache.
        """
        self.stats.requested += len(requests)
        results: list[Generation | None] = [None] * len(requests)
        keys: list[str] = []
        pending: list[int] = []
        for index, request in enumerate(requests):
            if self.cache is None:
                pending.append(index)
                continue
            key = self._key(request)
            keys.append(key)
            hit = self.cache.get(key)
            if hit is None:
                pending.append(index)
            else:
                results[index] = hit
        self.stats.cached += len(requests) - len(pending)
        if progress is not None and len(requests) > len(pending):
            progress(len(requests) - len(pending))

        semaphore = asyncio.Semaphore(self.backend.max_concurrency)
        size = self.backend.max_batch_size

        async def run_batch(indices: list[int]) -> None:
            async with semaphore:
                generations = await self._generate_with_retry([requests[i] for i in indices])
            for index, generation in zip(indices, generations, strict=True):
                results[index] = generation
                if self.cache is not None:
                    self.cache.put(keys[index], generation)
            self.stats.generated += len(indices)
            if progress is not None:
                progress(len(indices))

        batches = [pending[start : start + size] for start in range(0, len(pending), size)]
        await _run_fail_fast([run_batch(batch) for batch in batches])
        return [result for result in results if result is not None]

    async def _generate_with_retry(self, batch: list[GenerationRequest]) -> list[Generation]:
        attempt = 1
        while True:
            try:
                generations = await self.backend.generate_batch(batch)
            except RetryableBackendError as error:
                if attempt >= self.retry.max_attempts:
                    raise BackendError(
                        f"{self.backend.name}: giving up after {attempt} attempts: {error}"
                    ) from error
                delay = self.retry.delay(attempt, error.retry_after, self._rng)
                logger.warning(
                    "%s: %s; retrying in %.1fs (attempt %d of %d)",
                    self.backend.name,
                    error,
                    delay,
                    attempt + 1,
                    self.retry.max_attempts,
                )
                self.stats.retries += 1
                await self._sleep(delay)
                attempt += 1
                continue
            if len(generations) != len(batch):
                raise BackendError(
                    f"{self.backend.name}: returned {len(generations)} generations "
                    f"for {len(batch)} requests"
                )
            return generations


async def _run_fail_fast(coroutines: Sequence[Awaitable[None]]) -> None:
    """Run coroutines concurrently; on the first failure cancel the rest and re-raise."""
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    if not tasks:
        return
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            error = task.exception()
            if error is not None:
                raise error
    finally:
        unfinished = [task for task in tasks if not task.done()]
        for task in unfinished:
            task.cancel()
        if unfinished:
            await asyncio.gather(*unfinished, return_exceptions=True)
