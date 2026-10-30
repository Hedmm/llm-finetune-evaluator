"""Markdown report (GitHub-flavoured)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from llm_finetune_evaluator.analysis import Comparison, ExperimentAnalysis
from llm_finetune_evaluator.report import formatting as fmt


def _cell(text: Any) -> str:
    """Make ``text`` safe for a Markdown table cell."""
    value = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return value.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]], align: str = "") -> list[str]:
    alignment = align or "l" + "r" * (len(header) - 1)
    separators = ["---:" if a == "r" else "---" for a in alignment]
    lines = [
        "| " + " | ".join(_cell(h) for h in header) + " |",
        "| " + " | ".join(separators) + " |",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return lines


def _reference_text(reference: Any) -> str:
    return reference if isinstance(reference, str) else json.dumps(reference, ensure_ascii=False)


def _metric_label(analysis: ExperimentAnalysis, name: str) -> str:
    metric = analysis.metric(name)
    direction = "" if metric.higher_is_better else " (lower is better)"
    flag = " \u26a0 unreliable judge" if analysis.judge_unreliable(metric) else ""
    return f"{name}{direction}{flag}"


def _comparison_rows(
    analysis: ExperimentAnalysis, comparisons: Iterable[Comparison]
) -> list[list[str]]:
    rows = []
    for c in comparisons:
        rows.append(
            [
                _metric_label(analysis, c.metric),
                f"{c.candidate} vs {c.baseline}",
                c.slice_label,
                str(c.n),
                fmt.points(c.difference),
                fmt.interval(c.ci_low, c.ci_high, signed=True),
                fmt.p_value(c.p_value),
                fmt.p_value(c.p_adjusted),
                f"{c.candidate_better} / {c.baseline_better}",
                fmt.mde(c.mde),
                fmt.verdict(c),
            ]
        )
    return rows


_COMPARISON_HEADER = [
    "Metric",
    "Comparison",
    "Slice",
    "n",
    "Difference",
    "CI",
    "p",
    "p (Holm)",
    "Wins / losses",
    "MDE",
    "Verdict",
]


def render_markdown(analysis: ExperimentAnalysis) -> str:
    """Render the full report as Markdown."""
    settings = analysis.settings
    confidence = f"{settings.confidence:.0%}"
    lines: list[str] = [f"# {analysis.title}", ""]
    if analysis.description:
        lines += [analysis.description, ""]
    n_items = sum(int(eval_set.get("n_items", 0)) for eval_set in analysis.eval_sets)
    lines += [
        f"Generated {fmt.timestamp(analysis.created_at)} by llm-finetune-evaluator "
        f"{analysis.tool_version}. "
        f"Baseline: **{analysis.baseline}**. {len(analysis.models)} models, {n_items} items in "
        f"{len(analysis.eval_sets)} eval set(s).",
        "",
    ]
    for warning in analysis.warnings:
        lines += [f"> **Warning:** {warning}", ""]
    if analysis.judge_status == "uncalibrated":
        lines += [
            "> **Note:** judge-based metrics are not calibrated against human labels. "
            "Run `lfe judge-calibrate` before relying on them.",
            "",
        ]

    lines += ["## Headline metrics", ""]
    lines += [
        f"Mean score per model with {confidence} percentile bootstrap intervals "
        f"({settings.bootstrap_resamples:,} resamples, seed {settings.seed}).",
        "",
    ]
    header = ["Metric", "n", *analysis.models]
    rows = []
    for metric in analysis.metrics:
        per_model = analysis.headline[metric.name]
        n_values = {result.estimate.n for result in per_model.values() if result.estimate}
        rows.append(
            [
                _metric_label(analysis, metric.name),
                "/".join(str(n) for n in sorted(n_values)) or "0",
                *(fmt.estimate(per_model[model].estimate) for model in analysis.models),
            ]
        )
    lines += _table(header, rows)
    lines += ["", "Metric definitions:", ""]
    lines += [f"- `{metric.name}`: {metric.description}." for metric in analysis.metrics]
    lines.append("")

    if analysis.candidates:
        lines += ["## Comparison with the baseline", ""]
        lines += [
            "Paired tests on the items both models answered: exact McNemar for 0/1 scores, a "
            "paired permutation test otherwise. p-values are Holm-adjusted across all model pairs "
            "of each metric. MDE is the smallest difference this comparison would detect with "
            f"{settings.power:.0%} power at alpha = {settings.alpha}.",
            "",
        ]
        baseline_rows = [c for c in analysis.pairwise if c.baseline == analysis.baseline]
        lines += _table(
            _COMPARISON_HEADER, _comparison_rows(analysis, baseline_rows), "llllrrrrrrl"
        )
        lines.append("")
        others = [c for c in analysis.pairwise if c.baseline != analysis.baseline]
        if others:
            lines += ["### Other model pairs", ""]
            lines += _table(_COMPARISON_HEADER, _comparison_rows(analysis, others), "llllrrrrrrl")
            lines.append("")

        lines += ["## Regressions", ""]
        if analysis.regressions:
            lines += [
                "Comparisons where a candidate is significantly worse than the baseline after Holm "
                f"correction (alpha = {settings.alpha}):",
                "",
            ]
            lines += _table(
                _COMPARISON_HEADER, _comparison_rows(analysis, analysis.regressions), "llllrrrrrrl"
            )
        else:
            lines += [
                "No regressions: no candidate is significantly worse than the baseline, "
                "overall or in "
                "any slice, after Holm correction."
            ]
        lines.append("")
        if analysis.possible_regressions:
            lines += [
                "Possible regressions (significant before correction only; worth a closer look, "
                "not yet evidence):",
                "",
            ]
            lines += _table(
                _COMPARISON_HEADER,
                _comparison_rows(analysis, analysis.possible_regressions),
                "llllrrrrrrl",
            )
            lines.append("")

    lines += ["## Slices", ""]
    lines += [
        f"Every metric per slice (slices with at least {settings.min_slice_size} items). "
        "\u25b2 / \u25bc: the candidate is significantly better / worse than the baseline in that "
        "slice after Holm correction across the candidate's slices for the metric.",
        "",
    ]
    for metric in analysis.metrics:
        rows_for_metric = analysis.slices.get(metric.name, [])
        if not rows_for_metric:
            continue
        lines += [f"### {_metric_label(analysis, metric.name)}", ""]
        slice_rows = []
        for row in rows_for_metric:
            label = f"{row.dimension}={row.value}"
            if row.aliases:
                label += f" (same items as {', '.join(row.aliases)})"
            cells = [label, str(row.n)]
            for model in analysis.models:
                text = fmt.estimate(row.estimates.get(model))
                comparison = row.comparisons.get(model)
                if comparison is not None and comparison.outcome == "worse":
                    text += " \u25bc"
                elif comparison is not None and comparison.outcome == "better":
                    text += " \u25b2"
                cells.append(text)
            slice_rows.append(cells)
        lines += _table(["Slice", "n", *analysis.models], slice_rows)
        lines.append("")

    if analysis.examples:
        lines += ["## Wins and losses", ""]
        for examples in analysis.examples:
            lines += [
                f"### {examples.candidate} vs {analysis.baseline} on {examples.metric}",
                "",
                f"{examples.candidate} scores higher on {examples.n_wins} items and lower on "
                f"{examples.n_losses}. The largest differences:",
                "",
            ]
            for kind, example_rows in (("Win", examples.wins), ("Loss", examples.losses)):
                for example in example_rows:
                    tags = ", ".join(f"{k}={v}" for k, v in example.tags.items())
                    lines += [
                        f"**{kind}: `{example.id}`** ({example.task}{', ' + tags if tags else ''})",
                        "",
                        f"> {_cell(fmt.truncate(example.input, 400))}",
                        "",
                        "Reference: "
                        + _cell(fmt.truncate(_reference_text(example.reference), 200)),
                        "",
                    ]
                    lines += _table(
                        [
                            f"{analysis.baseline} (score {example.baseline_score:g})",
                            f"{examples.candidate} (score {example.candidate_score:g})",
                        ],
                        [
                            [
                                fmt.truncate(example.baseline_output),
                                fmt.truncate(example.candidate_output),
                            ]
                        ],
                        "ll",
                    )
                    lines.append("")

    label_rows = []
    for metric in analysis.metrics:
        for model in analysis.models:
            result = analysis.headline[metric.name][model]
            if result.labels:
                counts = ", ".join(f"{label}: {count}" for label, count in result.labels.items())
                label_rows.append([metric.name, model, counts])
    if label_rows:
        lines += ["## Outcome labels", ""]
        lines += [
            "How often each label occurred: refusals, made-up answers, invalid labels, "
            "judge verdicts.",
            "",
        ]
        lines += _table(["Metric", "Model", "Counts"], label_rows, "lll")
        lines.append("")

    if analysis.extraction:
        lines += ["## Extraction details", "", "Micro-averaged over all fields of all items.", ""]
        extraction_rows = [
            [
                metric,
                model,
                fmt.percent(values["precision"]),
                fmt.percent(values["recall"]),
                fmt.percent(values["f1"]),
                str(int(values["invalid_json"])),
            ]
            for metric, per_model in analysis.extraction.items()
            for model, values in per_model.items()
        ]
        lines += _table(
            ["Metric", "Model", "Precision", "Recall", "F1", "Invalid JSON"],
            extraction_rows,
            "llrrrr",
        )
        lines.append("")

    if analysis.judge_status != "not_used":
        lines += ["## LLM judge", ""]
        judge = analysis.judge or {}
        lines += [
            f"Judge: `{judge.get('name', 'judge')}` ({judge.get('kind', 'unknown backend')}).",
            "",
        ]
        calibration = analysis.calibration
        if calibration is None:
            lines += ["Not calibrated against human labels.", ""]
        else:
            status = "reliable" if calibration.reliable else "**unreliable**"
            lines += [
                f"Calibration on {calibration.n} human-labelled outputs: agreement "
                f"{fmt.percent(calibration.agreement)}, "
                f"Cohen's kappa {fmt.kappa(calibration.kappa)} "
                f"({calibration.confidence:.0%} CI {fmt.kappa(calibration.kappa_low)} to "
                f"{fmt.kappa(calibration.kappa_high)}), threshold {calibration.threshold:.2f}: "
                f"{status}.",
                "",
            ]
            matrix = calibration.confusion
            used = [
                (index, category)
                for index, category in enumerate(calibration.categories)
                if sum(matrix[index]) or any(row[index] for row in matrix)
            ]
            header = ["Human \\ judge", *(category for _, category in used)]
            matrix_rows = [
                [category, *(str(matrix[i][j]) for j, _ in used)] for i, category in used
            ]
            lines += _table(header, matrix_rows)
            lines.append("")

    lines += ["## Run metadata", ""]
    run_rows = []
    for run in analysis.runs:
        backend = run.backend
        model_id = backend.get("model") or backend.get("file") or backend.get("kind")
        sampling = ", ".join(f"{k}={v}" for k, v in run.sampling.items() if v not in (None, [], ()))
        run_rows.append(
            [
                run.model + (" (baseline)" if run.model == analysis.baseline else ""),
                str(backend.get("kind", "")),
                str(model_id),
                sampling,
                str(run.n_items),
                f"{run.generated} new / {run.cached} cached",
                fmt.timestamp(run.finished_at),
            ]
        )
    lines += _table(
        ["Model", "Backend", "Model id", "Sampling", "Items", "Generations", "Finished"],
        run_rows,
        "lllllll",
    )
    lines.append("")
    eval_rows = [
        [
            str(e.get("name")),
            str(e.get("path")),
            str(e.get("n_items")),
            str(e.get("sha256", ""))[:16],
        ]
        for e in analysis.eval_sets
    ]
    lines += _table(["Eval set", "File", "Items", "Content SHA-256 (prefix)"], eval_rows, "llrl")
    lines += [
        "",
        f"Statistics: {confidence} percentile bootstrap intervals "
        f"({settings.bootstrap_resamples:,} "
        f"resamples), exact McNemar and paired permutation tests "
        f"({settings.permutation_resamples:,} resamples when not exact), Holm correction, "
        f"alpha = {settings.alpha}, seed {settings.seed}.",
        "",
    ]
    return "\n".join(lines)
