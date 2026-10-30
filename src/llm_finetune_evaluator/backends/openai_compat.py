"""Backend for OpenAI-compatible HTTP APIs.

Works with any server that implements ``POST /v1/chat/completions`` (or the
older ``/v1/completions``): vLLM, the llama.cpp server, ollama, LM Studio, TGI
and OpenAI itself.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from llm_finetune_evaluator.backends.base import Backend, Generation, GenerationRequest
from llm_finetune_evaluator.errors import BackendError, RetryableBackendError

RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _message_text(content: Any) -> str:
    """Extract text from a chat message ``content`` field (string or parts list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


class OpenAICompatibleBackend(Backend):
    """A model served behind an OpenAI-compatible API.

    Args:
        name: Display name in reports.
        model: Model name sent to the server (for vLLM, the served model or the
            LoRA adapter name).
        base_url: API root, for example ``http://localhost:8000/v1``.
        api_key: Bearer token, or ``None`` for servers without authentication.
        api: ``"chat"`` for chat completions, ``"completions"`` for plain text
            completion (messages are joined into one prompt).
        timeout_s: Request timeout in seconds.
        max_concurrency: Maximum number of requests in flight.
        extra_body: Additional JSON fields sent with every request, such as
            vLLM's ``top_k`` or ``repetition_penalty``.
        cache_tag: Free-form label included in the cache key; change it when
            the weights behind an endpoint change but the model name does not.
        transport: Optional httpx transport, used by tests to mock the server.
    """

    kind = "openai"

    def __init__(
        self,
        name: str,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        api: str = "chat",
        timeout_s: float = 120.0,
        max_concurrency: int = 4,
        extra_body: dict[str, Any] | None = None,
        cache_tag: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name, max_concurrency=max_concurrency)
        if api not in ("chat", "completions"):
            raise ValueError(f"api must be 'chat' or 'completions', got {api!r}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api = api
        self.extra_body = dict(extra_body or {})
        self.cache_tag = cache_tag
        self.url = f"{self.base_url}/{'chat/completions' if api == 'chat' else 'completions'}"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
            limits=httpx.Limits(max_connections=max_concurrency),
        )

    def identity(self) -> dict[str, Any]:
        """Return the endpoint, model, API flavour, extra body and cache tag."""
        return {
            "backend": self.kind,
            "base_url": self.base_url,
            "model": self.model,
            "api": self.api,
            "extra_body": self.extra_body,
            "cache_tag": self.cache_tag,
        }

    def _body(self, request: GenerationRequest) -> dict[str, Any]:
        sampling = request.sampling
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
        }
        if sampling.seed is not None:
            body["seed"] = sampling.seed
        if sampling.stop:
            body["stop"] = list(sampling.stop)
        if self.api == "chat":
            body["messages"] = [message.model_dump() for message in request.messages]
        else:
            body["prompt"] = "\n\n".join(message.content for message in request.messages)
        body.update(self.extra_body)
        return body

    async def _generate_one(self, request: GenerationRequest) -> Generation:
        started = time.perf_counter()
        try:
            response = await self._client.post(self.url, json=self._body(request))
        except httpx.TimeoutException as error:
            raise RetryableBackendError(f"request to {self.url} timed out") from error
        except httpx.TransportError as error:
            raise RetryableBackendError(
                f"cannot reach {self.url} ({error.__class__.__name__}); is the server running?"
            ) from error
        latency = time.perf_counter() - started
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise RetryableBackendError(
                f"HTTP {response.status_code} from {self.url}", retry_after=_retry_after(response)
            )
        if response.status_code >= 400:
            raise BackendError(
                f"{self.name}: HTTP {response.status_code} from {self.url}: {response.text[:500]}"
            )
        try:
            data = response.json()
            choice = data["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise BackendError(
                f"{self.name}: unexpected response from {self.url}: {response.text[:500]}"
            ) from error
        if self.api == "chat":
            text = _message_text((choice.get("message") or {}).get("content"))
        else:
            text = str(choice.get("text") or "")
        usage = data.get("usage") or {}
        return Generation(
            text=text,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_s=round(latency, 4),
        )

    async def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        """Send each request in turn (the engine runs batches concurrently)."""
        return [await self._generate_one(request) for request in requests]

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        await self._client.aclose()
