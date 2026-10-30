"""Scorers turn a model output into a per-item score for one metric.

Built-in scorers:

* ``exact_match``, ``normalized_match`` and ``label_set`` for classification
  and short answers;
* ``json_fields`` for field-level precision, recall and F1 on JSON extraction;
* ``numeric`` for numbers within a tolerance;
* ``regex`` for pattern checks;
* ``refusal`` for needless refusals on answerable items;
* ``trap`` for trap and refusal_expected items;
* ``judge`` for rubric grading by a separate LLM judge.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llm_finetune_evaluator.backends.engine import GenerationEngine
from llm_finetune_evaluator.config import (
    ExactMatchConfig,
    ExperimentConfig,
    JsonFieldsConfig,
    JudgeScorerConfig,
    LabelSetConfig,
    NormalizedMatchConfig,
    NumericConfig,
    RefusalConfig,
    RegexConfig,
    ScorerConfig,
    TrapConfig,
)
from llm_finetune_evaluator.dataset import EvalSet
from llm_finetune_evaluator.errors import ConfigError
from llm_finetune_evaluator.scorers.base import MetricInfo, Score, Scorer, ScorerScope
from llm_finetune_evaluator.scorers.extraction import JsonFieldsScorer
from llm_finetune_evaluator.scorers.judge import JudgeScorer
from llm_finetune_evaluator.scorers.matching import (
    ExactMatchScorer,
    LabelSetScorer,
    NormalizedMatchScorer,
    RegexScorer,
)
from llm_finetune_evaluator.scorers.numeric import NumericScorer
from llm_finetune_evaluator.scorers.refusal import RefusalScorer, TrapScorer

__all__ = [
    "ExactMatchScorer",
    "JsonFieldsScorer",
    "JudgeScorer",
    "LabelSetScorer",
    "MetricInfo",
    "NormalizedMatchScorer",
    "NumericScorer",
    "RefusalScorer",
    "RegexScorer",
    "Score",
    "Scorer",
    "ScorerScope",
    "TrapScorer",
    "build_scorers",
    "read_labels_file",
]


def read_labels_file(path: Path) -> list[str]:
    """Read one label per line, ignoring blank lines and ``#`` comments.

    Raises:
        ConfigError: If the file does not exist or lists no labels.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ConfigError(f"labels file not found: {path}") from error
    labels = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not labels:
        raise ConfigError(f"labels file {path} lists no labels")
    return labels


def _infer_labels(scope: ScorerScope, eval_sets: Sequence[EvalSet]) -> list[str]:
    tasks = scope.tasks if scope.tasks is not None else LabelSetScorer.default_tasks
    labels = {
        str(reference)
        for eval_set in eval_sets
        if scope.eval_sets is None or eval_set.name in scope.eval_sets
        for item in eval_set.items
        if item.task in tasks
        for reference in item.references()
    }
    return sorted(labels)


def build_scorer(
    config: ScorerConfig,
    experiment: ExperimentConfig,
    eval_sets: Sequence[EvalSet],
    judge_engine: GenerationEngine | None = None,
) -> Scorer:
    """Build one scorer from its config.

    Args:
        config: The scorer's config entry.
        experiment: The whole experiment config (for the judge settings).
        eval_sets: Loaded eval sets, used to infer a label set when none is given.
        judge_engine: Engine of the judge model; required by ``judge`` scorers.

    Raises:
        ConfigError: If the scorer cannot be built from its settings.
    """
    name = config.resolved_name
    scope = ScorerScope.of(config.tasks, config.eval_sets)
    if isinstance(config, ExactMatchConfig):
        return ExactMatchScorer(name, case_sensitive=config.case_sensitive, scope=scope)
    if isinstance(config, NormalizedMatchConfig):
        return NormalizedMatchScorer(name, mode=config.mode, scope=scope)
    if isinstance(config, LabelSetConfig):
        if config.labels is not None:
            labels = list(config.labels)
        elif config.labels_file is not None:
            labels = read_labels_file(config.labels_file)
        else:
            labels = _infer_labels(scope, eval_sets)
        if not labels:
            raise ConfigError(f"scorer {name!r}: no labels given and no classification items found")
        return LabelSetScorer(name, labels=labels, multi_label=config.multi_label, scope=scope)
    if isinstance(config, JsonFieldsConfig):
        return JsonFieldsScorer(
            name,
            numeric_tolerance=config.numeric_tolerance,
            case_sensitive=config.case_sensitive,
            scope=scope,
        )
    if isinstance(config, NumericConfig):
        return NumericScorer(
            name,
            abs_tol=config.abs_tol,
            rel_tol=config.rel_tol,
            extract=config.extract,
            scope=scope,
        )
    if isinstance(config, RegexConfig):
        return RegexScorer(
            name,
            pattern=config.pattern,
            ignore_case=config.ignore_case,
            full_match=config.full_match,
            scope=scope,
        )
    if isinstance(config, RefusalConfig):
        return RefusalScorer(name, extra_patterns=config.extra_patterns, scope=scope)
    if isinstance(config, TrapConfig):
        return TrapScorer(name, extra_patterns=config.extra_patterns, scope=scope)
    if isinstance(config, JudgeScorerConfig):
        if judge_engine is None or experiment.judge is None:
            raise ConfigError(f"scorer {name!r} needs the judge model from the 'judge' section")
        return JudgeScorer(
            name,
            engine=judge_engine,
            sampling=experiment.judge.sampling,
            partial_credit=config.partial_credit,
            instructions=experiment.judge.instructions,
            scope=scope,
        )
    raise ConfigError(f"unsupported scorer {config!r}")


def build_scorers(
    experiment: ExperimentConfig,
    eval_sets: Sequence[EvalSet],
    judge_engine: GenerationEngine | None = None,
) -> list[Scorer]:
    """Build every scorer listed in the experiment config, in order."""
    return [
        build_scorer(config, experiment, eval_sets, judge_engine) for config in experiment.scorers
    ]
