"""The backend interface: generation requests, results and the abstract backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from llm_finetune_evaluator.config import SamplingParams
from llm_finetune_evaluator.dataset import ChatMessage


@dataclass(frozen=True)
class GenerationRequest:
    """One prompt to complete.

    Attributes:
        messages: The chat messages to send.
        sampling: Decoding parameters.
        item_id: Id of the eval item the request belongs to. Live backends
            ignore it; the mock and replay backends use it to look the item up.
    """

    messages: tuple[ChatMessage, ...]
    sampling: SamplingParams
    item_id: str | None = None

    def payload(self) -> dict[str, Any]:
        """Return the JSON-serialisable content that determines the output."""
        return {
            "messages": [message.model_dump() for message in self.messages],
            "sampling": self.sampling.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class Generation:
    """A completed generation.

    Attributes:
        text: Raw text returned by the model.
        finish_reason: Why generation stopped, if the backend reports it
            (for example ``"stop"`` or ``"length"``).
        prompt_tokens: Prompt length in tokens, if reported.
        completion_tokens: Completion length in tokens, if reported.
        latency_s: Wall-clock time of the original request in seconds.
        cached: Whether this result was served from the on-disk cache.
    """

    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (without the ``cached`` flag)."""
        data = asdict(self)
        del data["cached"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, cached: bool = False) -> Generation:
        """Rebuild a generation from :meth:`to_dict` output."""
        return cls(
            text=str(data["text"]),
            finish_reason=data.get("finish_reason"),
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            latency_s=data.get("latency_s"),
            cached=cached,
        )


class Backend(ABC):
    """A model that turns chat messages into text.

    Subclasses implement :meth:`identity` and :meth:`generate_batch`. Caching,
    retries and concurrency are handled uniformly for every backend by
    :class:`~llm_finetune_evaluator.backends.engine.GenerationEngine`.

    Args:
        name: Display name of the model in reports.
        max_concurrency: Maximum number of batches in flight at once.
        max_batch_size: Maximum number of requests passed to one
            :meth:`generate_batch` call.
        cacheable: Whether generations should be stored in the on-disk cache.
    """

    kind: str = "abstract"

    def __init__(
        self,
        name: str,
        *,
        max_concurrency: int = 4,
        max_batch_size: int = 1,
        cacheable: bool = True,
    ) -> None:
        if max_concurrency < 1 or max_batch_size < 1:
            raise ValueError("max_concurrency and max_batch_size must be at least 1")
        self.name = name
        self.max_concurrency = max_concurrency
        self.max_batch_size = max_batch_size
        self.cacheable = cacheable

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return the settings that determine this model's outputs.

        The identity is part of every cache key, so it must change whenever the
        outputs could change (a different model, adapter, endpoint or revision)
        and must never contain secrets such as API keys.
        """

    def request_identity(self, request: GenerationRequest) -> dict[str, Any]:
        """Return request fields beyond the prompt that affect the output.

        Live models only look at the prompt, so the default is empty. Backends
        that answer by item id override this.
        """
        return {}

    @abstractmethod
    async def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        """Generate one completion per request, in order.

        Raises:
            RetryableBackendError: For transient failures (rate limits,
                timeouts, server errors); the engine retries these.
            BackendError: For failures that retrying cannot fix.
        """

    async def aclose(self) -> None:  # noqa: B027 - optional hook with a no-op default
        """Release connections, models or other resources."""

    def describe(self) -> dict[str, Any]:
        """Return metadata about the backend for run records."""
        return {"name": self.name, "kind": self.kind, **self.identity()}
