"""Model backends and the machinery that runs them.

Every backend implements :class:`Backend`; :class:`GenerationEngine` adds the
on-disk cache, retries with backoff and bounded concurrency on top, the same way
for every backend.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from llm_finetune_evaluator.backends.base import Backend, Generation, GenerationRequest
from llm_finetune_evaluator.backends.cache import GenerationCache, default_cache_dir
from llm_finetune_evaluator.backends.engine import GenerationEngine, RetryPolicy
from llm_finetune_evaluator.config import (
    MockJudgeConfig,
    MockModelConfig,
    ModelConfig,
    OpenAIModelConfig,
    ReplayModelConfig,
    TransformersModelConfig,
)
from llm_finetune_evaluator.dataset import EvalItem
from llm_finetune_evaluator.errors import ConfigError

__all__ = [
    "Backend",
    "Generation",
    "GenerationCache",
    "GenerationEngine",
    "GenerationRequest",
    "RetryPolicy",
    "create_backend",
    "default_cache_dir",
]


def create_backend(config: ModelConfig, items: Mapping[str, EvalItem] | None = None) -> Backend:
    """Build the backend described by a model config.

    Args:
        config: A model (or judge) config.
        items: Eval items by id; required by the mock backend.

    Raises:
        ConfigError: If a required API key is missing.
    """
    if isinstance(config, OpenAIModelConfig):
        from llm_finetune_evaluator.backends.openai_compat import OpenAICompatibleBackend

        key_env = config.resolved_api_key_env()
        api_key = os.environ.get(key_env) if key_env else None
        if key_env and not api_key:
            raise ConfigError(
                f"model {config.name!r}: environment variable {key_env} with the API key is not set"
            )
        return OpenAICompatibleBackend(
            config.name,
            model=config.model,
            base_url=config.base_url,
            api_key=api_key,
            api=config.api,
            timeout_s=config.timeout_s,
            max_concurrency=config.max_concurrency,
            extra_body=config.extra_body,
            cache_tag=config.cache_tag,
        )
    if isinstance(config, TransformersModelConfig):
        from llm_finetune_evaluator.backends.huggingface import TransformersBackend

        return TransformersBackend(
            config.name,
            model=config.model,
            adapter=config.adapter,
            revision=config.revision,
            dtype=config.dtype,
            device=config.device,
            batch_size=config.batch_size,
            use_chat_template=config.use_chat_template,
            trust_remote_code=config.trust_remote_code,
            cache_tag=config.cache_tag,
        )
    if isinstance(config, MockModelConfig):
        from llm_finetune_evaluator.backends.mock import MockBackend

        return MockBackend(config, items or {})
    if isinstance(config, MockJudgeConfig):
        from llm_finetune_evaluator.backends.mock import MockJudgeBackend

        return MockJudgeBackend(config)
    if isinstance(config, ReplayModelConfig):
        from llm_finetune_evaluator.backends.replay import ReplayBackend

        return ReplayBackend(
            config.name,
            config.predictions,
            id_field=config.id_field,
            output_field=config.output_field,
        )
    raise ConfigError(f"unsupported backend {config!r}")
