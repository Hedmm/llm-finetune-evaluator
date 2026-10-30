"""Runs an experiment: generate with every model, score every output, write run directories."""

from __future__ import annotations

import asyncio
import hashlib
import platform
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_finetune_evaluator import __version__
from llm_finetune_evaluator.backends import create_backend
from llm_finetune_evaluator.backends.base import GenerationRequest
from llm_finetune_evaluator.backends.cache import GenerationCache
from llm_finetune_evaluator.backends.engine import GenerationEngine, ProgressCallback, RetryPolicy
from llm_finetune_evaluator.calibration import (
    CALIBRATION_FILE_NAME,
    CalibrationResult,
    calibrate_judge,
    load_human_labels,
    save_calibration,
)
from llm_finetune_evaluator.config import (
    ExperimentConfig,
    ModelConfig,
    OpenAIModelConfig,
    TransformersModelConfig,
)
from llm_finetune_evaluator.dataset import (
    ChatMessage,
    EvalItem,
    EvalSet,
    load_eval_set,
    render_messages,
)
from llm_finetune_evaluator.errors import ConfigError, DatasetError
from llm_finetune_evaluator.scorers import JudgeScorer, Scorer, build_scorers
from llm_finetune_evaluator.scorers.base import Score
from llm_finetune_evaluator.storage import (
    EXPERIMENT_FILE,
    ItemRecord,
    display_path,
    write_json,
    write_run,
)
from llm_finetune_evaluator.text import strip_think_tags

ProgressFactory = Callable[[str, int], ProgressCallback]
"""Creates a progress callback from a description and a total count."""

_MODEL_FAMILIES = (
    "gpt",
    "llama",
    "qwen",
    "mistral",
    "mixtral",
    "gemma",
    "phi",
    "deepseek",
    "falcon",
    "olmo",
    "granite",
    "command",
    "glm",
    "yi",
)


@dataclass(frozen=True)
class PreparedItem:
    """An eval item with its eval set name and the rendered prompt."""

    eval_set: str
    item: EvalItem
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True)
class RunResult:
    """What :func:`run_experiment` wrote.

    Attributes:
        output_dir: The experiment directory.
        runs: Run directory of every model that was run, by model name.
        calibration: Judge calibration, if the config asked for one.
        warnings: Problems worth showing to the user.
    """

    output_dir: Path
    runs: dict[str, Path]
    calibration: CalibrationResult | None
    warnings: tuple[str, ...]


def load_eval_sets(config: ExperimentConfig) -> list[EvalSet]:
    """Load every eval set of the config and check that ids are unique across them.

    Raises:
        DatasetError: If a file is invalid or an id occurs in two eval sets.
    """
    eval_sets = [load_eval_set(entry.path, entry.name) for entry in config.eval_sets]
    owner: dict[str, str] = {}
    for eval_set in eval_sets:
        for item in eval_set.items:
            if item.id in owner:
                raise DatasetError(
                    f"item id {item.id!r} occurs in eval sets {owner[item.id]!r} and "
                    f"{eval_set.name!r}; ids must be unique across eval sets"
                )
            owner[item.id] = eval_set.name
    return eval_sets


def prepare_items(config: ExperimentConfig, eval_sets: Sequence[EvalSet]) -> list[PreparedItem]:
    """Render the prompt of every item, so template errors surface before any model call."""
    prompts = {entry.name: entry.prompt for entry in config.eval_sets}
    return [
        PreparedItem(eval_set.name, item, render_messages(item, prompts[eval_set.name]))
        for eval_set in eval_sets
        for item in eval_set.items
    ]


def combined_hash(eval_sets: Sequence[EvalSet]) -> str:
    """Hash of every eval set's name and content; runs are comparable when it matches."""
    digest = hashlib.sha256()
    for eval_set in eval_sets:
        digest.update(f"{eval_set.name}\0{eval_set.content_hash}\n".encode())
    return digest.hexdigest()


def _model_identifier(model: ModelConfig) -> str | None:
    if isinstance(model, (OpenAIModelConfig, TransformersModelConfig)):
        return model.model
    return None


def _family(identifier: str) -> str | None:
    lowered = identifier.casefold()
    for family in _MODEL_FAMILIES:
        if re.search(rf"(?<![a-z]){family}", lowered):
            return family
    return None


def judge_family_warnings(config: ExperimentConfig) -> list[str]:
    """Warn when the judge looks like the same model or family as an evaluated model.

    Judges tend to rate outputs of their own model family higher, which is why
    the judge should come from a different family.
    """
    if config.judge is None:
        return []
    judge_id = _model_identifier(config.judge.model)
    if judge_id is None:
        return []
    judge_family = _family(judge_id)
    warnings = []
    for model in config.models:
        model_id = _model_identifier(model)
        if model_id is None:
            continue
        if model_id == judge_id:
            warnings.append(
                f"The judge ({judge_id}) is the same model as {model.name!r}; "
                "a model grading its own outputs is biased. Use a judge from another family."
            )
        elif judge_family is not None and judge_family == _family(model_id):
            warnings.append(
                f"The judge ({judge_id}) is from the same model family ({judge_family}) as "
                f"{model.name!r} ({model_id}); judges tend to prefer their own family. "
                "Use a judge from another family."
            )
    return warnings


def _sanitised_config(config: ExperimentConfig) -> dict[str, Any]:
    """Config snapshot for experiment.json with paths shown relative to the working directory."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return display_path(value)
        if isinstance(value, dict):
            return {key: convert(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(child) for child in value]
        return value

    data: dict[str, Any] = convert(config.model_dump(mode="python"))
    return data


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def _score(
    scorers: Sequence[Scorer],
    prepared: Sequence[PreparedItem],
    outputs: Sequence[str],
    model_name: str,
    progress: ProgressFactory | None,
) -> list[dict[str, Score]]:
    per_item: list[dict[str, Score]] = [{} for _ in prepared]
    for scorer in scorers:
        indices = [
            i for i, entry in enumerate(prepared) if scorer.applies_to(entry.item, entry.eval_set)
        ]
        if not indices:
            continue
        if isinstance(scorer, JudgeScorer):
            scorer.progress = progress(f"{model_name}: judge", len(indices)) if progress else None
        scores = await scorer.score_many(
            [prepared[i].item for i in indices], [outputs[i] for i in indices]
        )
        for index, score in zip(indices, scores, strict=True):
            per_item[index][scorer.name] = score
    return per_item


async def _run_model(
    model: ModelConfig,
    config: ExperimentConfig,
    prepared: Sequence[PreparedItem],
    items_by_id: Mapping[str, EvalItem],
    scorers: Sequence[Scorer],
    cache: GenerationCache | None,
    progress: ProgressFactory | None,
) -> tuple[dict[str, Any], list[ItemRecord]]:
    started = time.perf_counter()
    started_at = _now()
    sampling = model.sampling.apply(config.sampling)
    backend = create_backend(model, items_by_id)
    engine = GenerationEngine(backend, cache=cache, retry=RetryPolicy.from_config(config.retry))
    requests = [GenerationRequest(entry.messages, sampling, entry.item.id) for entry in prepared]
    try:
        callback = progress(f"{model.name}: generate", len(requests)) if progress else None
        generations = await engine.generate(requests, callback)
    finally:
        await backend.aclose()
    outputs = [strip_think_tags(g.text) if model.strip_think_tags else g.text for g in generations]
    scores = await _score(scorers, prepared, outputs, model.name, progress)
    records = [
        ItemRecord(
            id=entry.item.id,
            eval_set=entry.eval_set,
            task=entry.item.task,
            tags=dict(entry.item.tags),
            input=entry.item.input_text(),
            messages=entry.messages,
            reference=entry.item.reference,
            output=output,
            raw_output=generation.text if generation.text != output else None,
            finish_reason=generation.finish_reason,
            latency_s=generation.latency_s,
            cached=generation.cached,
            scores=item_scores,
        )
        for entry, generation, output, item_scores in zip(
            prepared, generations, outputs, scores, strict=True
        )
    ]
    meta = {
        "tool_version": __version__,
        "experiment": config.name,
        "model": model.name,
        "is_baseline": model.name == config.baseline_name,
        "backend": backend.describe(),
        "sampling": sampling.model_dump(mode="json"),
        "metrics": [scorer.info().to_dict() for scorer in scorers],
        "started_at": started_at,
        "finished_at": _now(),
        "duration_s": round(time.perf_counter() - started, 3),
        "generation": {
            "requested": engine.stats.requested,
            "cached": engine.stats.cached,
            "generated": engine.stats.generated,
            "retries": engine.stats.retries,
        },
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
    }
    return meta, records


async def _run(
    config: ExperimentConfig,
    output_dir: Path,
    model_names: Sequence[str] | None,
    use_cache: bool,
    config_path: Path | None,
    progress: ProgressFactory | None,
) -> RunResult:
    eval_sets = load_eval_sets(config)
    prepared = prepare_items(config, eval_sets)
    items_by_id = {entry.item.id: entry.item for entry in prepared}
    selected = list(config.models)
    if model_names:
        unknown = sorted(set(model_names) - {model.name for model in config.models})
        if unknown:
            raise ConfigError(
                f"unknown models {unknown}; the config defines {[m.name for m in config.models]}"
            )
        selected = [model for model in config.models if model.name in model_names]
    warnings = judge_family_warnings(config)
    cache = (
        GenerationCache.open_default(config.cache.dir)
        if use_cache and config.cache.enabled
        else None
    )
    judge_backend = create_backend(config.judge.model) if config.judge is not None else None
    judge_engine = (
        GenerationEngine(judge_backend, cache=cache, retry=RetryPolicy.from_config(config.retry))
        if judge_backend is not None
        else None
    )
    try:
        scorers = build_scorers(config, eval_sets, judge_engine)
        active = []
        for scorer in scorers:
            if any(scorer.applies_to(entry.item, entry.eval_set) for entry in prepared):
                active.append(scorer)
            else:
                warnings.append(f"Scorer {scorer.name!r} applies to no item and was skipped.")
        if not active:
            raise ConfigError(
                "no scorer applies to any item; check the scorers' tasks and eval_sets"
            )
        eval_set_meta = [
            {
                "name": eval_set.name,
                "path": display_path(eval_set.path),
                "sha256": eval_set.content_hash,
                "n_items": len(eval_set.items),
            }
            for eval_set in eval_sets
        ]
        eval_set_hash = combined_hash(eval_sets)
        runs: dict[str, Path] = {}
        for model in selected:
            meta, records = await _run_model(
                model, config, prepared, items_by_id, active, cache, progress
            )
            meta["eval_sets"] = eval_set_meta
            meta["eval_set_hash"] = eval_set_hash
            meta["model_index"] = [m.name for m in config.models].index(model.name)
            meta["judge"] = judge_backend.describe() if judge_backend is not None else None
            run_dir = output_dir / model.name
            write_run(run_dir, meta, records)
            runs[model.name] = run_dir
        calibration = None
        if config.judge is not None and config.judge.calibration is not None and judge_engine:
            judge_scorer = next((s for s in active if isinstance(s, JudgeScorer)), None)
            if judge_scorer is None:
                judge_scorer = JudgeScorer(
                    "judge_score",
                    engine=judge_engine,
                    sampling=config.judge.sampling,
                    instructions=config.judge.instructions,
                )
            judge_scorer.progress = None
            labels_path = config.judge.calibration.labels
            calibration = await calibrate_judge(
                judge_scorer,
                load_human_labels(labels_path),
                items_by_id,
                threshold=config.judge.calibration.kappa_threshold,
                seed=config.stats.seed,
                labels_file=labels_path,
            )
            save_calibration(calibration, output_dir / CALIBRATION_FILE_NAME)
            if calibration.warning:
                warnings.append(calibration.warning)
        write_json(
            output_dir / EXPERIMENT_FILE,
            {
                "format_version": 1,
                "tool_version": __version__,
                "name": config.name,
                "description": config.description,
                "updated_at": _now(),
                "config_file": display_path(config_path) if config_path else None,
                "baseline": config.baseline_name,
                "models": [model.name for model in config.models],
                "eval_sets": eval_set_meta,
                "eval_set_hash": eval_set_hash,
                "judge": judge_backend.describe() if judge_backend is not None else None,
                "warnings": warnings,
                "stats": config.stats.model_dump(mode="json"),
                "report": config.report.model_dump(mode="json"),
                "config": _sanitised_config(config),
            },
        )
        return RunResult(output_dir, runs, calibration, tuple(warnings))
    finally:
        if judge_backend is not None:
            await judge_backend.aclose()
        if cache is not None:
            cache.close()


def run_experiment(
    config: ExperimentConfig,
    *,
    output_dir: Path | None = None,
    models: Sequence[str] | None = None,
    use_cache: bool = True,
    config_path: Path | None = None,
    progress: ProgressFactory | None = None,
) -> RunResult:
    """Run every model of the experiment on every eval set and score the outputs.

    Args:
        config: The experiment config.
        output_dir: Where to write runs; defaults to the config's output directory.
        models: Run only these models (others keep their previous runs).
        use_cache: Read and write the generation cache.
        config_path: Path of the config file, recorded in the metadata.
        progress: Optional factory for progress callbacks.

    Returns:
        The directories written and any warnings.
    """
    directory = output_dir if output_dir is not None else config.resolved_output_dir()
    return asyncio.run(_run(config, directory, models, use_cache, config_path, progress))
