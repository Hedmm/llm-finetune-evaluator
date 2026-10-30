"""Number formatting shared by the Markdown and HTML reports."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from llm_finetune_evaluator.analysis import Comparison, Estimate


def percent(value: float | None) -> str:
    """Format a share as a percentage with one decimal ("72.4%")."""
    if value is None or math.isnan(value):
        return "n/a"
    return f"{100 * value:.1f}%"


def points(value: float | None, signed: bool = True) -> str:
    """Format a difference of shares in percentage points ("+5.2 pts")."""
    if value is None or math.isnan(value):
        return "n/a"
    return f"{100 * value:+.1f} pts" if signed else f"{100 * value:.1f} pts"


def interval(low: float | None, high: float | None, signed: bool = False) -> str:
    """Format an interval of shares ("68.1 to 76.5")."""
    if low is None or high is None or math.isnan(low) or math.isnan(high):
        return "n/a"
    if signed:
        return f"{100 * low:+.1f} to {100 * high:+.1f}"
    return f"{100 * low:.1f} to {100 * high:.1f}"


def estimate(value: Estimate | None) -> str:
    """Format an estimate with its interval ("72.4% (68.1 to 76.5)")."""
    if value is None:
        return "n/a"
    return f"{percent(value.value)} ({interval(value.low, value.high)})"


def p_value(value: float) -> str:
    """Format a p-value: three decimals, scientific notation below 0.001."""
    if value <= 0.0:
        return "< 1e-300"
    if value >= 0.001:
        return f"{value:.3f}"
    return f"{value:.1e}"


def mde(value: float | None) -> str:
    """Format a minimum detectable effect ("±4.1 pts")."""
    if value is None or math.isnan(value):
        return "n/a"
    return f"±{100 * value:.1f} pts"


def kappa(value: float) -> str:
    """Format Cohen's kappa with two decimals."""
    return "undefined" if math.isnan(value) else f"{value:.2f}"


def verdict(comparison: Comparison) -> str:
    """Short text verdict with a direction symbol, for tables."""
    if comparison.outcome == "better":
        return "\u25b2 better"
    if comparison.outcome == "worse":
        return "\u25bc worse"
    return "not significant"


def timestamp(value: str | None) -> str:
    """Format an ISO 8601 timestamp as "2026-10-30 14:57 UTC"."""
    if not value:
        return ""
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return value
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def truncate(text: str, limit: int = 600) -> str:
    """Shorten long texts for display, marking the cut."""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
