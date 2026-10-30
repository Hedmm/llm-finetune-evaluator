"""Statistical comparison of runs: headline metrics, pairwise tests, slices and regressions.

The analysis works on run directories, so it can be repeated with other
settings without calling any model. Everything is computed on paired items:
an item counts for a comparison only if both models have a score for it.

Multiple comparisons are corrected with Holm's method in two families:

* the pairwise table: all model pairs, separately for each metric;
* the slice tables: all slices of one candidate against the baseline,
  separately for each metric.

A regression is a comparison against the baseline that is significantly worse
after correction. Slices that are worse before correction only are listed as
possible regressions: worth a look, not yet evidence.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np

from llm_finetune_evaluator import __version__
from llm_finetune_evaluator.calibration import CalibrationResult
from llm_finetune_evaluator.errors import RunError
from llm_finetune_evaluator.scorers.base import MetricInfo
from llm_finetune_evaluator.stats import (
    bootstrap_ci,
    derive_generator,
    holm_bonferroni,
    mcnemar_exact,
    minimum_detectable_effect,
    minimum_detectable_effect_continuous,
    paired_bootstrap_ci,
    paired_permutation_test,
)
from llm_finetune_evaluator.storage import ItemRecord, RunData

MCNEMAR = "exact McNemar"
PERMUTATION = "paired permutation"


@dataclass(frozen=True)
class AnalysisSettings:
    """Settings of one analysis; defaults match the config defaults."""

    alpha: float = 0.05
    confidence: float = 0.95
    bootstrap_resamples: int = 10_000
    permutation_resamples: int = 10_000
    seed: int = 0
    min_slice_size: int = 10
    slice_by: tuple[str, ...] | None = None
    power: float = 0.8
    max_examples: int = 6
    primary_metric: str | None = None

    @classmethod
    def from_experiment(
        cls, experiment: Mapping[str, Any] | None, **overrides: Any
    ) -> AnalysisSettings:
        """Build settings from ``experiment.json`` (if any) and explicit overrides."""
        stats = dict((experiment or {}).get("stats") or {})
        report = dict((experiment or {}).get("report") or {})
        values: dict[str, Any] = {}
        for key in (
            "alpha",
            "confidence",
            "bootstrap_resamples",
            "permutation_resamples",
            "seed",
            "min_slice_size",
            "power",
        ):
            if key in stats:
                values[key] = stats[key]
        if stats.get("slice_by") is not None:
            values["slice_by"] = tuple(stats["slice_by"])
        if "max_examples" in report:
            values["max_examples"] = report["max_examples"]
        if report.get("primary_metric") is not None:
            values["primary_metric"] = report["primary_metric"]
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)


@dataclass(frozen=True)
class Estimate:
    """A mean with its percentile bootstrap interval."""

    value: float
    low: float
    high: float
    n: int


@dataclass(frozen=True)
class ModelMetric:
    """One model's result on one metric.

    Attributes:
        model: Model name.
        estimate: Mean and interval, or ``None`` when no item was scored.
        labels: How often each score label occurred.
        n_unscored: Items the metric applies to that could not be scored.
    """

    model: str
    estimate: Estimate | None
    labels: dict[str, int]
    n_unscored: int


@dataclass(frozen=True)
class Comparison:
    """A paired comparison of a candidate with a reference model.

    ``difference`` is ``candidate - baseline`` on the paired items; whether a
    positive difference is good depends on ``higher_is_better``.
    """

    metric: str
    baseline: str
    candidate: str
    dimension: str | None
    value: str | None
    n: int
    baseline_mean: float
    candidate_mean: float
    difference: float
    ci_low: float
    ci_high: float
    test: str
    p_value: float
    p_adjusted: float
    family_size: int
    candidate_better: int
    baseline_better: int
    mde: float | None
    higher_is_better: bool
    alpha: float

    @property
    def significant(self) -> bool:
        """Whether the Holm-adjusted p-value is below alpha."""
        return self.p_adjusted < self.alpha

    @property
    def improvement(self) -> float:
        """The difference signed so that positive always means better."""
        return self.difference if self.higher_is_better else -self.difference

    @property
    def outcome(self) -> str:
        """``"better"``, ``"worse"`` or ``"no significant difference"`` for the candidate."""
        if self.significant and self.improvement > 0:
            return "better"
        if self.significant and self.improvement < 0:
            return "worse"
        return "no significant difference"

    @property
    def slice_label(self) -> str:
        """``"all items"`` or ``"dimension=value"``."""
        return "all items" if self.dimension is None else f"{self.dimension}={self.value}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict including the derived fields."""
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "slice": self.slice_label,
            "n": self.n,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "difference": self.difference,
            "ci": [self.ci_low, self.ci_high],
            "test": self.test,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "family_size": self.family_size,
            "candidate_better": self.candidate_better,
            "baseline_better": self.baseline_better,
            "mde": self.mde,
            "significant": self.significant,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class SliceRow:
    """One slice of one metric: every model's estimate and each candidate's comparison.

    ``aliases`` lists other slices (``"dimension=value"``) that contain exactly
    the same items for this metric; they are tested once, under this name.
    """

    dimension: str
    value: str
    n: int
    estimates: dict[str, Estimate | None]
    comparisons: dict[str, Comparison]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExampleRow:
    """An item where the candidate and the baseline scored differently."""

    id: str
    eval_set: str
    task: str
    tags: dict[str, str]
    input: str
    reference: Any
    baseline_output: str
    candidate_output: str
    baseline_score: float
    candidate_score: float
    baseline_label: str | None
    candidate_label: str | None


@dataclass(frozen=True)
class Examples:
    """Wins and losses of one candidate against the baseline on the primary metric."""

    candidate: str
    metric: str
    n_wins: int
    n_losses: int
    wins: list[ExampleRow]
    losses: list[ExampleRow]


@dataclass(frozen=True)
class RunSummary:
    """Metadata of one run for the report."""

    model: str
    is_baseline: bool
    backend: dict[str, Any]
    sampling: dict[str, Any]
    n_items: int
    generated: int
    cached: int
    retries: int
    duration_s: float | None
    started_at: str | None
    finished_at: str | None
    path: str


@dataclass(frozen=True)
class ExperimentAnalysis:
    """Everything a report shows."""

    title: str
    experiment: str
    description: str | None
    created_at: str
    tool_version: str
    models: list[str]
    baseline: str
    metrics: list[MetricInfo]
    settings: AnalysisSettings
    headline: dict[str, dict[str, ModelMetric]]
    pairwise: list[Comparison]
    slices: dict[str, list[SliceRow]]
    slice_dimensions: list[str]
    regressions: list[Comparison]
    possible_regressions: list[Comparison]
    examples: list[Examples]
    extraction: dict[str, dict[str, dict[str, float]]]
    calibration: CalibrationResult | None
    judge_status: str
    judge: dict[str, Any] | None
    eval_sets: list[dict[str, Any]]
    runs: list[RunSummary]
    warnings: list[str] = field(default_factory=list)

    @property
    def candidates(self) -> list[str]:
        """Every model except the baseline."""
        return [model for model in self.models if model != self.baseline]

    def metric(self, name: str) -> MetricInfo:
        """Look up a metric by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)

    def comparisons_against_baseline(self, metric: str) -> list[Comparison]:
        """Headline comparisons of every candidate with the baseline on ``metric``."""
        return [c for c in self.pairwise if c.metric == metric and c.baseline == self.baseline]

    def judge_unreliable(self, metric: MetricInfo) -> bool:
        """Whether ``metric`` comes from a judge that failed calibration."""
        return metric.uses_judge and self.judge_status == "unreliable"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (``analysis.json``)."""

        def estimate(value: Estimate | None) -> dict[str, Any] | None:
            if value is None:
                return None
            return {"value": value.value, "ci": [value.low, value.high], "n": value.n}

        summary: dict[str, Any] = _finite(
            {
                "title": self.title,
                "experiment": self.experiment,
                "created_at": self.created_at,
                "tool_version": self.tool_version,
                "models": self.models,
                "baseline": self.baseline,
                "settings": {
                    "alpha": self.settings.alpha,
                    "confidence": self.settings.confidence,
                    "bootstrap_resamples": self.settings.bootstrap_resamples,
                    "permutation_resamples": self.settings.permutation_resamples,
                    "seed": self.settings.seed,
                    "min_slice_size": self.settings.min_slice_size,
                    "power": self.settings.power,
                },
                "metrics": [metric.to_dict() for metric in self.metrics],
                "headline": {
                    metric: {
                        model: {
                            "estimate": estimate(result.estimate),
                            "labels": result.labels,
                            "n_unscored": result.n_unscored,
                        }
                        for model, result in per_model.items()
                    }
                    for metric, per_model in self.headline.items()
                },
                "pairwise": [comparison.to_dict() for comparison in self.pairwise],
                "regressions": [comparison.to_dict() for comparison in self.regressions],
                "possible_regressions": [c.to_dict() for c in self.possible_regressions],
                "slices": {
                    metric: [
                        {
                            "slice": f"{row.dimension}={row.value}",
                            "aliases": list(row.aliases),
                            "n": row.n,
                            "estimates": {m: estimate(e) for m, e in row.estimates.items()},
                            "comparisons": {m: c.to_dict() for m, c in row.comparisons.items()},
                        }
                        for row in rows
                    ]
                    for metric, rows in self.slices.items()
                },
                "extraction": self.extraction,
                "judge_status": self.judge_status,
                "calibration": self.calibration.to_dict() if self.calibration else None,
                "eval_sets": self.eval_sets,
                "warnings": self.warnings,
            }
        )
        return summary


def _finite(value: Any) -> Any:
    """Replace NaN and infinities with ``None`` so the result is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(child) for child in value]
    return value


def _metric_values(run: RunData, metric: str) -> tuple[dict[str, float], Counter[str], int]:
    values: dict[str, float] = {}
    labels: Counter[str] = Counter()
    unscored = 0
    for record in run.records:
        score = record.scores.get(metric)
        if score is None:
            continue
        if score.label is not None:
            labels[score.label] += 1
        if score.value is None:
            unscored += 1
        else:
            values[record.id] = score.value
    return values, labels, unscored


def _estimate(values: Sequence[float], settings: AnalysisSettings, *key: str) -> Estimate | None:
    if not values:
        return None
    interval = bootstrap_ci(
        values,
        confidence=settings.confidence,
        n_resamples=settings.bootstrap_resamples,
        seed=derive_generator(settings.seed, "estimate", *key),
    )
    return Estimate(interval.estimate, interval.low, interval.high, len(values))


def _compare(
    metric: MetricInfo,
    baseline: str,
    candidate: str,
    baseline_values: Mapping[str, float],
    candidate_values: Mapping[str, float],
    ids: Sequence[str],
    settings: AnalysisSettings,
    dimension: str | None = None,
    value: str | None = None,
) -> Comparison | None:
    paired = [
        item_id for item_id in ids if item_id in baseline_values and item_id in candidate_values
    ]
    if not paired:
        return None
    a = np.array([candidate_values[item_id] for item_id in paired])
    b = np.array([baseline_values[item_id] for item_id in paired])
    key = (metric.name, baseline, candidate, dimension or "", value or "")
    interval = paired_bootstrap_ci(
        a,
        b,
        confidence=settings.confidence,
        n_resamples=settings.bootstrap_resamples,
        seed=derive_generator(settings.seed, "difference", *key),
    )
    n = len(paired)
    binary = metric.binary and bool(
        np.all(np.isin(a, (0.0, 1.0))) and np.all(np.isin(b, (0.0, 1.0)))
    )
    mde: float | None
    if binary:
        result = mcnemar_exact(a, b)
        test, p_value = MCNEMAR, result.p_value
        candidate_better, baseline_better = result.a_only, result.b_only
        discordance = result.n_discordant / n
        mde = (
            minimum_detectable_effect(n, discordance, alpha=settings.alpha, power=settings.power)
            if discordance > 0
            else None
        )
    else:
        permutation = paired_permutation_test(
            a,
            b,
            n_resamples=settings.permutation_resamples,
            seed=derive_generator(settings.seed, "permutation", *key),
        )
        test, p_value = PERMUTATION, permutation.p_value
        candidate_better, baseline_better = int(np.sum(a > b)), int(np.sum(a < b))
        spread = float(np.std(a - b, ddof=1)) if n > 1 else 0.0
        mde = (
            minimum_detectable_effect_continuous(
                n, spread, alpha=settings.alpha, power=settings.power
            )
            if spread > 0
            else None
        )
    return Comparison(
        metric=metric.name,
        baseline=baseline,
        candidate=candidate,
        dimension=dimension,
        value=value,
        n=n,
        baseline_mean=float(b.mean()),
        candidate_mean=float(a.mean()),
        difference=interval.estimate,
        ci_low=interval.low,
        ci_high=interval.high,
        test=test,
        p_value=p_value,
        p_adjusted=p_value,
        family_size=1,
        candidate_better=candidate_better,
        baseline_better=baseline_better,
        mde=mde,
        higher_is_better=metric.higher_is_better,
        alpha=settings.alpha,
    )


def _adjust(comparisons: list[Comparison]) -> list[Comparison]:
    """Apply Holm's correction to one family of comparisons."""
    adjusted = holm_bonferroni([comparison.p_value for comparison in comparisons])
    return [
        replace(comparison, p_adjusted=p, family_size=len(comparisons))
        for comparison, p in zip(comparisons, adjusted, strict=True)
    ]


def _slice_value(record: ItemRecord, dimension: str) -> str | None:
    if dimension == "task":
        return record.task.value
    if dimension == "eval_set":
        return record.eval_set
    return record.tags.get(dimension)


def _slice_dimensions(
    runs: Sequence[RunData], settings: AnalysisSettings
) -> tuple[list[str], list[str]]:
    tags = sorted({key for record in runs[0].records for key in record.tags})
    automatic = ["task"]
    if len({record.eval_set for record in runs[0].records}) > 1:
        automatic.append("eval_set")
    available = [*automatic, *tags]
    if settings.slice_by is None:
        return available, []
    unknown = [dimension for dimension in settings.slice_by if dimension not in available]
    warnings = [
        f"Slice dimension {d!r} does not occur in the data and was ignored." for d in unknown
    ]
    return [d for d in settings.slice_by if d in available], warnings


def _resolve_baseline(
    runs: Sequence[RunData], baseline: str | None, experiment: Mapping[str, Any] | None
) -> str:
    names = [run.model for run in runs]
    if baseline is not None:
        if baseline not in names:
            raise RunError(f"baseline {baseline!r} is not among the runs {names}")
        return baseline
    configured = (experiment or {}).get("baseline")
    if configured in names:
        return str(configured)
    flagged = [run.model for run in runs if run.meta.get("is_baseline")]
    if len(flagged) == 1:
        return flagged[0]
    return names[0]


def _check_compatible(runs: Sequence[RunData]) -> None:
    names = [run.model for run in runs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RunError(f"several runs have the same model name: {duplicates}")
    hashes = {run.eval_set_hash for run in runs}
    if len(hashes) > 1:
        detail = ", ".join(f"{run.model}: {run.eval_set_hash[:12]}" for run in runs)
        raise RunError(
            "the runs were scored on different eval set contents, so they cannot be paired "
            f"({detail}). Re-run them on the same eval sets; cached generations make that cheap."
        )


def _run_summary(run: RunData) -> RunSummary:
    generation = run.meta.get("generation") or {}
    return RunSummary(
        model=run.model,
        is_baseline=bool(run.meta.get("is_baseline")),
        backend=dict(run.meta.get("backend") or {}),
        sampling=dict(run.meta.get("sampling") or {}),
        n_items=len(run.records),
        generated=int(generation.get("generated", 0)),
        cached=int(generation.get("cached", 0)),
        retries=int(generation.get("retries", 0)),
        duration_s=run.meta.get("duration_s"),
        started_at=run.meta.get("started_at"),
        finished_at=run.meta.get("finished_at"),
        path=run.path.as_posix(),
    )


def _examples(
    metric: MetricInfo,
    baseline: RunData,
    candidate: RunData,
    limit: int,
    seed: int,
) -> Examples:
    base_records = baseline.by_id()
    rows: list[tuple[float, ExampleRow]] = []
    for record in candidate.records:
        other = base_records.get(record.id)
        if other is None:
            continue
        candidate_score = record.scores.get(metric.name)
        baseline_score = other.scores.get(metric.name)
        if candidate_score is None or baseline_score is None:
            continue
        if candidate_score.value is None or baseline_score.value is None:
            continue
        difference = candidate_score.value - baseline_score.value
        if difference == 0:
            continue
        improvement = difference if metric.higher_is_better else -difference
        rows.append(
            (
                improvement,
                ExampleRow(
                    id=record.id,
                    eval_set=record.eval_set,
                    task=record.task.value,
                    tags=record.tags,
                    input=record.input,
                    reference=record.reference,
                    baseline_output=other.output,
                    candidate_output=record.output,
                    baseline_score=baseline_score.value,
                    candidate_score=candidate_score.value,
                    baseline_label=baseline_score.label,
                    candidate_label=candidate_score.label,
                ),
            )
        )

    def order(row: tuple[float, ExampleRow]) -> tuple[float, str]:
        # Largest differences first; ties in a seeded but random-looking order, so
        # that the examples are not all from whichever eval set sorts first.
        tiebreak = hashlib.sha256(f"{seed}:{row[1].id}".encode()).hexdigest()
        return (-abs(row[0]), tiebreak)

    wins = sorted((r for r in rows if r[0] > 0), key=order)
    losses = sorted((r for r in rows if r[0] < 0), key=order)
    return Examples(
        candidate=candidate.model,
        metric=metric.name,
        n_wins=len(wins),
        n_losses=len(losses),
        wins=[row for _, row in wins[:limit]],
        losses=[row for _, row in losses[:limit]],
    )


def _extraction_summary(run: RunData, metric: str) -> dict[str, float]:
    tp = fp = fn = invalid = 0
    for record in run.records:
        score = record.scores.get(metric)
        if score is None:
            continue
        tp += int(score.details.get("tp", 0))
        fp += int(score.details.get("fp", 0))
        fn += int(score.details.get("fn", 0))
        invalid += score.label == "invalid_json"
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "invalid_json": float(invalid)}


def analyze(
    runs: Sequence[RunData],
    *,
    baseline: str | None = None,
    settings: AnalysisSettings | None = None,
    calibration: CalibrationResult | None = None,
    experiment: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> ExperimentAnalysis:
    """Compare runs on paired items.

    Args:
        runs: One run per model, all scored on the same eval set content.
        baseline: Baseline model; defaults to the experiment's baseline, then
            to the run flagged as baseline, then to the first run.
        settings: Statistical settings.
        calibration: Judge calibration to report and to mark judge metrics.
        experiment: Contents of ``experiment.json``, if available.
        title: Report title.

    Raises:
        RunError: If the runs cannot be compared.
    """
    if not runs:
        raise RunError("no runs to analyse")
    settings = settings or AnalysisSettings()
    _check_compatible(runs)
    baseline_name = _resolve_baseline(runs, baseline, experiment)
    ordered = sorted(
        runs,
        key=lambda run: (
            run.model != baseline_name,
            int(run.meta.get("model_index", 10_000)),
            run.model,
        ),
    )
    base_run = ordered[0]
    models = [run.model for run in ordered]
    warnings = list((experiment or {}).get("warnings") or [])

    metrics = base_run.metrics
    for run in ordered[1:]:
        missing = {m.name for m in metrics} - {m.name for m in run.metrics}
        if missing:
            warnings.append(f"Run {run.model!r} has no scores for {sorted(missing)}.")

    values: dict[str, dict[str, dict[str, float]]] = {}
    headline: dict[str, dict[str, ModelMetric]] = {}
    for metric in metrics:
        values[metric.name] = {}
        headline[metric.name] = {}
        for run in ordered:
            per_item, labels, unscored = _metric_values(run, metric.name)
            values[metric.name][run.model] = per_item
            headline[metric.name][run.model] = ModelMetric(
                model=run.model,
                estimate=_estimate(list(per_item.values()), settings, metric.name, run.model),
                labels=dict(sorted(labels.items())),
                n_unscored=unscored,
            )

    record_order = [record.id for record in base_run.records]
    pairwise: list[Comparison] = []
    for metric in metrics:
        family = []
        for i, first in enumerate(models):
            for second in models[i + 1 :]:
                comparison = _compare(
                    metric,
                    first,
                    second,
                    values[metric.name][first],
                    values[metric.name][second],
                    record_order,
                    settings,
                )
                if comparison is not None:
                    family.append(comparison)
        pairwise.extend(_adjust(family) if family else [])

    dimensions, slice_warnings = _slice_dimensions(ordered, settings)
    warnings.extend(slice_warnings)
    groups: dict[tuple[str, str], list[str]] = {}
    for record in base_run.records:
        for dimension in dimensions:
            slice_value = _slice_value(record, dimension)
            if slice_value is not None:
                groups.setdefault((dimension, slice_value), []).append(record.id)

    slices: dict[str, list[SliceRow]] = {}
    slice_comparisons: list[Comparison] = []
    for metric in metrics:
        base_values = values[metric.name][baseline_name]
        scored = [item_id for item_id in record_order if item_id in base_values]
        ordered_groups = sorted(
            groups.items(), key=lambda entry: (dimensions.index(entry[0][0]), entry[0][1])
        )
        selected: list[tuple[tuple[str, str], list[str]]] = []
        aliases: dict[tuple[str, str], list[str]] = {}
        seen: dict[frozenset[str], tuple[str, str]] = {}
        for key, ids in ordered_groups:
            members = [item_id for item_id in ids if item_id in base_values]
            if len(members) < settings.min_slice_size or len(members) == len(scored):
                continue
            signature = frozenset(members)
            if signature in seen:
                aliases[seen[signature]].append(f"{key[0]}={key[1]}")
                continue
            seen[signature] = key
            aliases[key] = []
            selected.append((key, members))
        per_candidate: dict[str, dict[tuple[str, str], Comparison]] = {}
        for candidate in models[1:]:
            family = [
                comparison
                for (dimension, slice_value), ids in selected
                if (
                    comparison := _compare(
                        metric,
                        baseline_name,
                        candidate,
                        base_values,
                        values[metric.name][candidate],
                        ids,
                        settings,
                        dimension,
                        slice_value,
                    )
                )
                is not None
            ]
            adjusted = _adjust(family) if family else []
            slice_comparisons.extend(adjusted)
            per_candidate[candidate] = {(c.dimension or "", c.value or ""): c for c in adjusted}
        rows = []
        for (dimension, slice_value), ids in selected:
            estimates = {
                model: _estimate(
                    [values[metric.name][model][i] for i in ids if i in values[metric.name][model]],
                    settings,
                    metric.name,
                    model,
                    dimension,
                    slice_value,
                )
                for model in models
            }
            comparisons = {
                candidate: per_candidate[candidate][(dimension, slice_value)]
                for candidate in models[1:]
                if (dimension, slice_value) in per_candidate[candidate]
            }
            rows.append(
                SliceRow(
                    dimension,
                    slice_value,
                    len(ids),
                    estimates,
                    comparisons,
                    tuple(aliases[(dimension, slice_value)]),
                )
            )
        slices[metric.name] = rows

    against_baseline = [c for c in pairwise if c.baseline == baseline_name]
    regressions = sorted(
        (c for c in [*against_baseline, *slice_comparisons] if c.outcome == "worse"),
        key=lambda c: (c.p_adjusted, c.improvement),
    )
    possible = sorted(
        (
            c
            for c in slice_comparisons
            if not c.significant and c.p_value < settings.alpha and c.improvement < 0
        ),
        key=lambda c: (c.p_value, c.improvement),
    )

    primary_name = settings.primary_metric or (metrics[0].name if metrics else None)
    examples = []
    if primary_name is not None and settings.max_examples > 0:
        primary = next((m for m in metrics if m.name == primary_name), None)
        if primary is None:
            warnings.append(f"Primary metric {primary_name!r} was not scored; no examples shown.")
        else:
            examples = [
                _examples(primary, base_run, run, settings.max_examples, settings.seed)
                for run in ordered[1:]
            ]

    extraction = {
        metric.name: {run.model: _extraction_summary(run, metric.name) for run in ordered}
        for metric in metrics
        if metric.scorer == "json_fields"
    }

    judge_metrics = [metric for metric in metrics if metric.uses_judge]
    if not judge_metrics:
        judge_status = "not_used"
    elif calibration is None:
        judge_status = "uncalibrated"
    else:
        judge_status = "reliable" if calibration.reliable else "unreliable"
    judge_warning = calibration.warning if calibration is not None else None
    if judge_status == "unreliable" and judge_warning and judge_warning not in warnings:
        warnings.append(judge_warning)

    experiment_name = str((experiment or {}).get("name") or base_run.meta.get("experiment") or "")
    configured_title = ((experiment or {}).get("report") or {}).get("title")
    default_title = (
        f"Evaluation report: {experiment_name}" if experiment_name else "Evaluation report"
    )
    return ExperimentAnalysis(
        title=title or configured_title or default_title,
        experiment=experiment_name,
        description=(experiment or {}).get("description"),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tool_version=__version__,
        models=models,
        baseline=baseline_name,
        metrics=metrics,
        settings=settings,
        headline=headline,
        pairwise=pairwise,
        slices=slices,
        slice_dimensions=dimensions,
        regressions=regressions,
        possible_regressions=possible,
        examples=examples,
        extraction=extraction,
        calibration=calibration,
        judge_status=judge_status,
        judge=base_run.meta.get("judge"),
        eval_sets=list(base_run.meta.get("eval_sets") or []),
        runs=[_run_summary(run) for run in ordered],
        warnings=warnings,
    )
