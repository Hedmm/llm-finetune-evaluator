"""Backend for local Hugging Face transformers models, with optional PEFT adapters.

Install the extra dependencies with ``pip install "llm-finetune-evaluator[transformers]"``.
Imports of torch, transformers and peft happen on first use, so the rest of the
package works without them.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_finetune_evaluator.backends.base import Backend, Generation, GenerationRequest
from llm_finetune_evaluator.config import SamplingParams
from llm_finetune_evaluator.dataset import ChatMessage
from llm_finetune_evaluator.errors import BackendError


def path_fingerprint(location: str) -> str | None:
    """Fingerprint a local model or adapter directory, or return ``None`` for hub ids.

    The fingerprint hashes the relative path, size and modification time of
    every file, which is cheap even for large checkpoints and changes whenever
    a checkpoint is retrained in place.
    """
    path = Path(location).expanduser()
    if not path.exists():
        return None
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file in files:
        stat = file.stat()
        relative = file.name if file == path else file.relative_to(path).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _truncate_at_stop(text: str, stop: Sequence[str]) -> tuple[str, bool]:
    positions = [text.find(marker) for marker in stop if marker and marker in text]
    if not positions:
        return text, False
    return text[: min(positions)], True


class TransformersBackend(Backend):
    """A causal language model loaded with transformers, optionally with a LoRA adapter.

    Args:
        name: Display name in reports.
        model: Hub id or local path of the base model.
        adapter: Hub id or local path of a PEFT adapter to load on top.
        revision: Hub revision (branch, tag or commit) of the base model.
        dtype: Weight dtype; ``"auto"`` picks bfloat16 on GPUs that support it,
            float16 on other GPUs and float32 on CPU.
        device: Torch device; defaults to CUDA, then MPS, then CPU.
        batch_size: Prompts per ``generate`` call (left-padded).
        use_chat_template: Render messages with the tokenizer's chat template;
            set to ``False`` for base models without one.
        trust_remote_code: Passed to ``from_pretrained``.
        cache_tag: Free-form label included in the cache key.
    """

    kind = "transformers"

    def __init__(
        self,
        name: str,
        *,
        model: str,
        adapter: str | None = None,
        revision: str | None = None,
        dtype: str = "auto",
        device: str | None = None,
        batch_size: int = 8,
        use_chat_template: bool = True,
        trust_remote_code: bool = False,
        cache_tag: str | None = None,
    ) -> None:
        super().__init__(name, max_concurrency=1, max_batch_size=batch_size)
        self.model_id = model
        self.adapter = adapter
        self.revision = revision
        self.dtype = dtype
        self.device = device
        self.use_chat_template = use_chat_template
        self.trust_remote_code = trust_remote_code
        self._identity = {
            "backend": self.kind,
            "model": model,
            "model_fingerprint": path_fingerprint(model),
            "revision": revision,
            "adapter": adapter,
            "adapter_fingerprint": path_fingerprint(adapter) if adapter else None,
            "dtype": dtype,
            "use_chat_template": use_chat_template,
            "cache_tag": cache_tag,
        }
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._lock = threading.Lock()

    def identity(self) -> dict[str, Any]:
        """Return the model, adapter, their fingerprints, revision and dtype."""
        return dict(self._identity)

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise BackendError(
                "the transformers backend needs extra packages: "
                'pip install "llm-finetune-evaluator[transformers]"'
            ) from error
        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if self.dtype == "auto":
            if device.startswith("cuda"):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = getattr(torch, self.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if self.adapter:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise BackendError(
                    "loading an adapter needs peft: "
                    'pip install "llm-finetune-evaluator[transformers]"'
                ) from error
            model = PeftModel.from_pretrained(model, self.adapter)
        model.to(device)
        model.eval()
        self._torch, self._tokenizer, self._model = torch, tokenizer, model
        self.device = device

    def _render(self, messages: Sequence[ChatMessage]) -> str:
        if not self.use_chat_template:
            return "\n\n".join(message.content for message in messages)
        if not getattr(self._tokenizer, "chat_template", None):
            raise BackendError(
                f"{self.name}: the tokenizer of {self.model_id} has no chat template; "
                "set use_chat_template: false for base models"
            )
        try:
            rendered = self._tokenizer.apply_chat_template(
                [message.model_dump() for message in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as error:
            raise BackendError(
                f"{self.name}: the chat template rejected the messages ({error}); "
                "some templates do not accept a system message"
            ) from error
        return str(rendered)

    def _generation_config(self, sampling: SamplingParams) -> Any:
        config = copy.deepcopy(self._model.generation_config)
        config.max_new_tokens = sampling.max_tokens
        config.pad_token_id = self._tokenizer.pad_token_id
        config.do_sample = sampling.temperature > 0
        if config.do_sample:
            config.temperature = sampling.temperature
            config.top_p = sampling.top_p
        else:
            config.temperature = None
            config.top_p = None
            config.top_k = None
        return config

    def _generate_group(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        torch, tokenizer, model = self._torch, self._tokenizer, self._model
        sampling = requests[0].sampling
        prompts = [self._render(request.messages) for request in requests]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=not self.use_chat_template,
        ).to(self.device)
        if sampling.seed is not None:
            torch.manual_seed(sampling.seed)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, generation_config=self._generation_config(sampling))
        latency = (time.perf_counter() - started) / len(requests)
        prompt_length = encoded["input_ids"].shape[1]
        stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
        generations = []
        for row, mask in zip(
            output[:, prompt_length:].tolist(), encoded["attention_mask"], strict=True
        ):
            length = next((i for i, token in enumerate(row) if token in stop_ids), len(row))
            text = tokenizer.decode(row[:length], skip_special_tokens=True)
            text, stopped = _truncate_at_stop(text, sampling.stop)
            truncated = length >= sampling.max_tokens and not stopped
            generations.append(
                Generation(
                    text=text,
                    finish_reason="length" if truncated else "stop",
                    prompt_tokens=int(mask.sum()),
                    completion_tokens=length,
                    latency_s=round(latency, 4),
                )
            )
        return generations

    def _generate_sync(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        with self._lock:
            if self._model is None:
                self._load()
            results: dict[int, Generation] = {}
            groups: dict[SamplingParams, list[int]] = {}
            for index, request in enumerate(requests):
                groups.setdefault(request.sampling, []).append(index)
            for indices in groups.values():
                generated = self._generate_group([requests[i] for i in indices])
                results.update(zip(indices, generated, strict=True))
            return [results[index] for index in range(len(requests))]

    async def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        """Generate a left-padded batch in a worker thread."""
        return await asyncio.to_thread(self._generate_sync, requests)

    async def aclose(self) -> None:
        """Drop the model so its memory can be released."""
        self._model = None
        self._tokenizer = None
