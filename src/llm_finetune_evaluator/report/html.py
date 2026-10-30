"""Self-contained HTML report: one file, inline CSS and SVG, no external resources."""

from __future__ import annotations

import json
from typing import Any

import jinja2

from llm_finetune_evaluator.analysis import Comparison, ExperimentAnalysis, RunSummary
from llm_finetune_evaluator.report import formatting as fmt
from llm_finetune_evaluator.report.charts import forest_chart, headline_chart, series_class

_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("llm_finetune_evaluator.report", "templates"),
    autoescape=True,
    undefined=jinja2.StrictUndefined,
    trim_blocks=False,
    lstrip_blocks=False,
)
_ENVIRONMENT.filters.update(
    {
        "percent": fmt.percent,
        "points": fmt.points,
        "signed_interval": lambda low, high: fmt.interval(low, high, signed=True),
        "pvalue": fmt.p_value,
        "mde": fmt.mde,
        "estimate": fmt.estimate,
        "kappa": fmt.kappa,
        "truncate_text": fmt.truncate,
        "timestamp": fmt.timestamp,
        "reference": lambda value: (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        ),
    }
)


def _model_id(run: RunSummary) -> str:
    backend = run.backend
    for key in ("model", "file", "kind"):
        if backend.get(key):
            return str(backend[key])
    return ""


def _sampling_text(run: RunSummary) -> str:
    return ", ".join(
        f"{key}={value}" for key, value in run.sampling.items() if value not in (None, [], ())
    )


def _summary(analysis: ExperimentAnalysis) -> list[dict[str, Any]]:
    if not analysis.metrics or not analysis.candidates:
        return []
    primary = analysis.settings.primary_metric or analysis.metrics[0].name
    cards = []
    for candidate in analysis.candidates:
        comparison: Comparison | None = next(
            (
                c
                for c in analysis.pairwise
                if c.metric == primary
                and c.baseline == analysis.baseline
                and c.candidate == candidate
            ),
            None,
        )
        if comparison is None:
            continue
        cards.append(
            {
                "candidate": candidate,
                "metric": primary,
                "comparison": comparison,
                "regressions": sum(1 for r in analysis.regressions if r.candidate == candidate),
            }
        )
    return cards


def render_html(analysis: ExperimentAnalysis) -> str:
    """Render the full report as one self-contained HTML document."""

    def n_for(metric: str) -> str:
        counts = sorted(
            {r.estimate.n for r in analysis.headline[metric].values() if r.estimate is not None}
        )
        return "/".join(str(n) for n in counts) or "0"

    def forest(candidate: str, metric: str) -> str | None:
        overall = next(
            (
                c
                for c in analysis.pairwise
                if c.metric == metric
                and c.baseline == analysis.baseline
                and c.candidate == candidate
            ),
            None,
        )
        return forest_chart(candidate, metric, analysis.slices.get(metric, []), overall)

    label_rows = [
        (
            metric.name,
            model,
            ", ".join(f"{label}: {count}" for label, count in result.labels.items()),
        )
        for metric in analysis.metrics
        for model in analysis.models
        if (result := analysis.headline[metric.name][model]).labels
    ]
    used_categories: list[tuple[int, str]] = []
    if analysis.calibration is not None:
        matrix = analysis.calibration.confusion
        used_categories = [
            (index, category)
            for index, category in enumerate(analysis.calibration.categories)
            if sum(matrix[index]) or any(row[index] for row in matrix)
        ]
    template = _ENVIRONMENT.get_template("report.html.j2")
    return template.render(
        a=analysis,
        s=analysis.settings,
        confidence=f"{analysis.settings.confidence:.0%}",
        n_items=sum(int(e.get("n_items", 0)) for e in analysis.eval_sets),
        summary=_summary(analysis),
        headline_chart=headline_chart(analysis),
        series_class=series_class,
        n_for=n_for,
        forest=forest,
        label_rows=label_rows,
        baseline_pairs=[c for c in analysis.pairwise if c.baseline == analysis.baseline],
        other_pairs=[c for c in analysis.pairwise if c.baseline != analysis.baseline],
        used_categories=used_categories,
        model_id=_model_id,
        sampling_text=_sampling_text,
    )
