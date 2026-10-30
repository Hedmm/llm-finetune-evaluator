"""Inline SVG charts for the HTML report (no JavaScript libraries, no external files).

Two forms, both "point estimate with interval" charts on one shared axis:

* :func:`headline_chart`: every model's estimate and confidence interval for
  every metric (dot and whisker), one colour per model in a fixed order.
* :func:`forest_chart`: per-slice differences between a candidate and the
  baseline, with a zero line. Significant regressions and improvements use the
  reserved status colours and always carry a text label as well.

Colours come from CSS custom properties defined in the report stylesheet, so
light and dark mode switch in one place. Every mark has an accessible name and
a ``data-tip`` text that the report's small tooltip script shows on hover and
keyboard focus; the tables next to each chart hold the same numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from llm_finetune_evaluator.analysis import Comparison, ExperimentAnalysis, SliceRow
from llm_finetune_evaluator.report.formatting import estimate, interval, p_value, points

SERIES_SLOTS = 8
"""Number of categorical colours; further models are drawn in the muted colour."""

_ROW = 22
_GROUP_GAP = 12
_LEFT = 190
_RIGHT = 120
_TOP = 28
_BOTTOM = 34
_WIDTH = 760


@dataclass(frozen=True)
class _Mark:
    label: str
    value: float
    low: float
    high: float
    css_class: str
    tip: str
    note: str = ""


def series_class(index: int) -> str:
    """CSS class of the categorical colour for model number ``index``."""
    return f"series-{index + 1}" if index < SERIES_SLOTS else "series-muted"


def _nice_step(span: float) -> float:
    raw = span / 5
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10.0 * magnitude


def _ticks(low: float, high: float) -> list[float]:
    step = _nice_step(high - low)
    start = math.ceil(low / step - 1e-9) * step
    ticks = []
    value = start
    while value <= high + 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _render(
    marks: Sequence[_Mark | None],
    domain: tuple[float, float],
    tick_format: str,
    title: str,
    zero_line: bool,
) -> str:
    """Draw horizontal interval marks, one per row; ``None`` adds a group gap."""
    rows = [mark for mark in marks if mark is not None]
    gaps = len(marks) - len(rows)
    height = _TOP + _BOTTOM + len(rows) * _ROW + gaps * _GROUP_GAP
    plot_width = _WIDTH - _LEFT - _RIGHT
    low, high = domain

    def x(value: float) -> float:
        clamped = min(max(value, low), high)
        return _LEFT + (clamped - low) / (high - low) * plot_width

    parts = [
        f'<svg class="chart" viewBox="0 0 {_WIDTH} {height}" role="group" '
        f'aria-label="{escape(title)}">'
    ]
    axis_y = height - _BOTTOM + 6
    for tick in _ticks(low, high):
        tick_x = x(tick)
        label = f"{100 * tick:+.0f}" if tick_format == "points" else f"{100 * tick:.0f}%"
        if tick_format == "points" and abs(tick) < 1e-12:
            label = "0"
        parts.append(
            f'<line class="grid" x1="{tick_x:.1f}" y1="{_TOP - 8}" '
            f'x2="{tick_x:.1f}" y2="{axis_y}"/>'
            f'<text class="tick" x="{tick_x:.1f}" y="{axis_y + 16}" '
            f'text-anchor="middle">{label}</text>'
        )
    if zero_line and low < 0 < high:
        parts.append(
            f'<line class="zero" x1="{x(0):.1f}" y1="{_TOP - 8}" x2="{x(0):.1f}" y2="{axis_y}"/>'
        )
    y = _TOP
    for mark in marks:
        if mark is None:
            y += _GROUP_GAP
            continue
        center = y + _ROW / 2
        name = escape(mark.tip)
        parts.append(
            f'<g class="mark {mark.css_class}" tabindex="0" role="img" aria-label="{name}" '
            f'data-tip="{name}">'
            f'<rect class="hit" x="0" y="{y}" width="{_WIDTH}" height="{_ROW}"/>'
            f'<text class="row-label" x="{_LEFT - 12}" y="{center + 4}" text-anchor="end">'
            f"{escape(mark.label)}</text>"
            f'<line class="whisker" x1="{x(mark.low):.1f}" y1="{center}" '
            f'x2="{x(mark.high):.1f}" y2="{center}"/>'
            f'<circle class="dot" cx="{x(mark.value):.1f}" cy="{center}" r="5"/>'
        )
        if mark.note:
            parts.append(
                f'<text class="note" x="{_WIDTH - _RIGHT + 12}" y="{center + 4}">'
                f"{escape(mark.note)}</text>"
            )
        parts.append("</g>")
        y += _ROW
    parts.append("</svg>")
    return "".join(parts)


def headline_chart(analysis: ExperimentAnalysis) -> str:
    """Dot-and-whisker chart of every model on every metric (0 to 100%)."""
    marks: list[_Mark | None] = []
    for metric_index, metric in enumerate(analysis.metrics):
        if metric_index:
            marks.append(None)
        for model_index, model in enumerate(analysis.models):
            result = analysis.headline[metric.name][model].estimate
            if result is None:
                continue
            label = metric.name if model_index == 0 else ""
            direction = "higher is better" if metric.higher_is_better else "lower is better"
            marks.append(
                _Mark(
                    label=label,
                    value=result.value,
                    low=result.low,
                    high=result.high,
                    css_class=series_class(model_index),
                    tip=(
                        f"{model} · {metric.name}: {estimate(result)}, n = {result.n} ({direction})"
                    ),
                )
            )
    return _render(
        marks, (0.0, 1.0), "percent", "Headline metrics with confidence intervals", False
    )


def _comparison_mark(label: str, comparison: Comparison) -> _Mark:
    if comparison.outcome == "worse":
        css_class, note = "status-critical", "▼ regression"
    elif comparison.outcome == "better":
        css_class, note = "status-good", "▲ better"
    else:
        css_class, note = "neutral", ""
    span = interval(comparison.ci_low, comparison.ci_high, signed=True)
    tip = (
        f"{comparison.candidate} vs {comparison.baseline} · {label}: "
        f"{points(comparison.difference)} ({span}), "
        f"n = {comparison.n}, Holm p = {p_value(comparison.p_adjusted)}"
    )
    return _Mark(
        label=label,
        value=comparison.difference,
        low=comparison.ci_low,
        high=comparison.ci_high,
        css_class=css_class,
        tip=tip,
        note=note,
    )


def forest_chart(
    candidate: str,
    metric_name: str,
    rows: Sequence[SliceRow],
    overall: Comparison | None,
) -> str | None:
    """Per-slice differences (candidate minus baseline) with intervals and a zero line.

    Returns ``None`` when the candidate has no slice comparison for the metric.
    """
    comparisons = [(f"{row.dimension}={row.value}", row.comparisons.get(candidate)) for row in rows]
    present = [(label, comparison) for label, comparison in comparisons if comparison is not None]
    if not present:
        return None
    marks: list[_Mark | None] = []
    if overall is not None:
        marks.extend([_comparison_mark("all items", overall), None])
    marks.extend(_comparison_mark(label, comparison) for label, comparison in present)
    extent = max(
        0.05,
        *(abs(mark.low) for mark in marks if mark is not None),
        *(abs(mark.high) for mark in marks if mark is not None),
    )
    bound = min(1.0, math.ceil(extent * 20) / 20)
    title = f"{candidate} minus baseline on {metric_name}, by slice"
    return _render(marks, (-bound, bound), "points", title, True)
