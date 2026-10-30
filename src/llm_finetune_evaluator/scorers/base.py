"""Scorer interface and the per-item score record."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from llm_finetune_evaluator.dataset import EvalItem, TaskType


@dataclass(frozen=True)
class Score:
    """The score of one output.

    Attributes:
        value: A number in [0, 1], or ``None`` when the output could not be
            scored (for example an unparseable judge reply). Unscored items are
            left out of that metric's statistics for every model.
        label: Optional category, such as ``"refused"``, ``"made_up"`` or a
            judge verdict. Reports show the distribution of labels.
        details: Scorer-specific diagnostics (parsed label, field counts, ...).
    """

    value: float | None
    label: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        data: dict[str, Any] = {"value": self.value}
        if self.label is not None:
            data["label"] = self.label
        if self.details:
            data["details"] = self.details
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Score:
        """Rebuild a score from :meth:`to_dict` output."""
        value = data.get("value")
        return cls(
            value=None if value is None else float(value),
            label=data.get("label"),
            details=dict(data.get("details") or {}),
        )


@dataclass(frozen=True)
class MetricInfo:
    """Describes the metric a scorer produces; stored with every run.

    Attributes:
        name: Metric name, unique within an experiment.
        scorer: Scorer type, for example ``"label_set"``.
        description: One-line explanation for reports.
        higher_is_better: Direction of improvement. Needless refusals, for
            instance, should go down.
        binary: Whether every value is 0 or 1. Binary metrics are compared with
            the exact McNemar test, others with a paired permutation test.
        tasks: Task types the metric covers.
        eval_sets: Eval sets the metric is restricted to, or ``None`` for all.
        uses_judge: Whether the metric comes from the LLM judge.
    """

    name: str
    scorer: str
    description: str
    higher_is_better: bool
    binary: bool
    tasks: tuple[str, ...]
    eval_sets: tuple[str, ...] | None = None
    uses_judge: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "name": self.name,
            "scorer": self.scorer,
            "description": self.description,
            "higher_is_better": self.higher_is_better,
            "binary": self.binary,
            "tasks": list(self.tasks),
            "eval_sets": list(self.eval_sets) if self.eval_sets is not None else None,
            "uses_judge": self.uses_judge,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricInfo:
        """Rebuild metric metadata from :meth:`to_dict` output."""
        eval_sets = data.get("eval_sets")
        return cls(
            name=data["name"],
            scorer=data["scorer"],
            description=data["description"],
            higher_is_better=bool(data["higher_is_better"]),
            binary=bool(data["binary"]),
            tasks=tuple(data["tasks"]),
            eval_sets=tuple(eval_sets) if eval_sets is not None else None,
            uses_judge=bool(data.get("uses_judge", False)),
        )


@dataclass(frozen=True)
class ScorerScope:
    """Which items a scorer applies to.

    Attributes:
        tasks: Task types to score; ``None`` uses the scorer's default tasks.
        eval_sets: Eval sets to score; ``None`` means all of them.
    """

    tasks: frozenset[TaskType] | None = None
    eval_sets: frozenset[str] | None = None

    @classmethod
    def of(
        cls, tasks: Iterable[TaskType] | None = None, eval_sets: Iterable[str] | None = None
    ) -> ScorerScope:
        """Build a scope from optional iterables."""
        return cls(
            tasks=frozenset(tasks) if tasks is not None else None,
            eval_sets=frozenset(eval_sets) if eval_sets is not None else None,
        )


class Scorer(ABC):
    """Scores model outputs for one metric.

    Subclasses set the class attributes and implement :meth:`score`; scorers
    that call a model (the judge) override :meth:`score_many` instead.

    Args:
        name: Metric name.
        scope: Restricts the scorer to some tasks or eval sets.
    """

    type_name: ClassVar[str]
    default_tasks: ClassVar[frozenset[TaskType]]
    higher_is_better: ClassVar[bool] = True
    uses_judge: ClassVar[bool] = False

    def __init__(self, name: str, scope: ScorerScope | None = None) -> None:
        scope = scope or ScorerScope()
        self.name = name
        self.tasks = scope.tasks if scope.tasks is not None else self.default_tasks
        self.eval_sets = scope.eval_sets

    @property
    def binary(self) -> bool:
        """Whether every score is 0 or 1."""
        return True

    @property
    def description(self) -> str:
        """One-line explanation of the metric for reports."""
        return (
            self.__class__.__doc__.strip().splitlines()[0] if self.__class__.__doc__ else self.name
        )

    def applies_to(self, item: EvalItem, eval_set: str) -> bool:
        """Whether this scorer scores ``item`` from ``eval_set``."""
        if item.task not in self.tasks:
            return False
        return self.eval_sets is None or eval_set in self.eval_sets

    def score(self, item: EvalItem, output: str) -> Score:
        """Score one output.

        Raises:
            ScorerError: If the item's reference cannot be used by this scorer.
        """
        raise NotImplementedError(f"{self.__class__.__name__} scores items with score_many")

    async def score_many(self, items: Sequence[EvalItem], outputs: Sequence[str]) -> list[Score]:
        """Score many outputs; the default calls :meth:`score` for each one."""
        return [self.score(item, output) for item, output in zip(items, outputs, strict=True)]

    def info(self) -> MetricInfo:
        """Return the metric metadata stored with each run."""
        return MetricInfo(
            name=self.name,
            scorer=self.type_name,
            description=self.description,
            higher_is_better=self.higher_is_better,
            binary=self.binary,
            tasks=tuple(sorted(task.value for task in self.tasks)),
            eval_sets=tuple(sorted(self.eval_sets)) if self.eval_sets is not None else None,
            uses_judge=self.uses_judge,
        )


class SyncScorer(Scorer):
    """A scorer that works on one output at a time without calling a model."""

    @abstractmethod
    def score(self, item: EvalItem, output: str) -> Score:
        """Score one output."""
