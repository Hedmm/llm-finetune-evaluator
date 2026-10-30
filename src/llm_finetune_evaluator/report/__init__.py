"""Markdown, HTML and JSON reports built from an experiment analysis."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from llm_finetune_evaluator.analysis import ExperimentAnalysis
from llm_finetune_evaluator.report.html import render_html
from llm_finetune_evaluator.report.markdown import render_markdown

REPORT_FORMATS = ("md", "html", "json")
"""Report formats: Markdown, self-contained HTML and machine-readable JSON."""

_FILE_NAMES = {"md": "report.md", "html": "report.html", "json": "analysis.json"}

__all__ = ["REPORT_FORMATS", "render_html", "render_markdown", "write_reports"]


def write_reports(
    analysis: ExperimentAnalysis, directory: Path, formats: Iterable[str] = REPORT_FORMATS
) -> dict[str, Path]:
    """Write the requested report formats into ``directory``.

    Returns:
        The written files by format.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for report_format in formats:
        if report_format not in _FILE_NAMES:
            raise ValueError(f"unknown report format {report_format!r}; use {REPORT_FORMATS}")
        path = directory / _FILE_NAMES[report_format]
        if report_format == "md":
            content = render_markdown(analysis)
        elif report_format == "html":
            content = render_html(analysis)
        else:
            content = json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False) + "\n"
        path.write_text(content, encoding="utf-8")
        written[report_format] = path
    return written
