from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from helpers import make_run
from llm_finetune_evaluator.analysis import AnalysisSettings, ExperimentAnalysis, analyze
from llm_finetune_evaluator.calibration import calibrate
from llm_finetune_evaluator.report import (
    REPORT_FORMATS,
    render_html,
    render_markdown,
    write_reports,
)
from llm_finetune_evaluator.report import formatting as fmt
from llm_finetune_evaluator.report.charts import forest_chart, headline_chart
from llm_finetune_evaluator.rubric import JudgeVerdict
from llm_finetune_evaluator.scorers.base import MetricInfo
from llm_finetune_evaluator.storage import RunData

SETTINGS = AnalysisSettings(bootstrap_resamples=500, permutation_resamples=500)
ACCURACY = MetricInfo("accuracy", "exact_match", "Exact match", True, True, ("open_qa",))
JUDGE = MetricInfo(
    "judge_score", "judge", "Judge score", True, False, ("open_qa",), uses_judge=True
)


def _runs(tmp_path: Path) -> list[RunData]:
    rng = np.random.default_rng(0)
    ids = [f"q{i:03d}" for i in range(160)]
    tags = {item_id: {"topic": "loans" if i % 4 == 0 else "cards"} for i, item_id in enumerate(ids)}
    base = {item_id: float(rng.random() < 0.7) for item_id in ids}
    tuned = {
        item_id: float(rng.random() < (0.1 if tags[item_id]["topic"] == "loans" else 0.95))
        for item_id in ids
    }
    metrics = [ACCURACY, JUDGE]
    return [
        make_run(tmp_path, "base", base, tags=tags, metrics=metrics, is_baseline=True),
        make_run(tmp_path, "tuned", tuned, tags=tags, metrics=metrics, model_index=1),
    ]


@pytest.fixture
def analysis(tmp_path: Path) -> ExperimentAnalysis:
    return analyze(
        _runs(tmp_path),
        settings=SETTINGS,
        experiment={"name": "unit", "report": {"title": "Unit report"}},
    )


def test_writes_every_format(tmp_path: Path, analysis: ExperimentAnalysis) -> None:
    written = write_reports(analysis, tmp_path / "out")
    assert set(written) == set(REPORT_FORMATS)
    assert written["md"].name == "report.md"
    assert written["html"].name == "report.html"
    assert json.loads(written["json"].read_text())["title"] == "Unit report"
    with pytest.raises(ValueError, match="unknown report format"):
        write_reports(analysis, tmp_path / "out", ["pdf"])


def test_html_is_self_contained(analysis: ExperimentAnalysis) -> None:
    html = render_html(analysis)
    assert html.startswith("<!doctype html>")
    assert "<link" not in html
    assert "@import" not in html
    assert not re.search(r"<script[^>]*\bsrc=", html)
    assert not re.search(r"https?://", html)
    assert not re.search(r"""(?:src|href)=["'](?!#)""", html)


def test_html_has_every_section(analysis: ExperimentAnalysis) -> None:
    html = render_html(analysis)
    for section in (
        "headline",
        "comparisons",
        "regressions",
        "slices",
        "examples",
        "judge",
        "metadata",
        "method",
    ):
        assert f'id="{section}"' in html
    assert "<title>Unit report</title>" in html
    assert html.count('<svg class="chart"') >= 3
    assert "topic=<strong>loans</strong>" in html
    assert "regression" in html


def test_regressions_are_listed(analysis: ExperimentAnalysis) -> None:
    assert any(r.slice_label == "topic=loans" for r in analysis.regressions)
    markdown = render_markdown(analysis)
    assert "## Regressions" in markdown
    assert re.search(r"\| accuracy \| tuned vs base \| topic=loans \| 40 \|", markdown)
    assert "▼ worse" in markdown


def test_untrusted_outputs_are_escaped(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    tuned = runs[1]
    hostile = "<script>alert(1)</script> | pipe"
    records = tuple(
        record.__class__(**{**record.__dict__, "output": hostile}) for record in tuned.records
    )
    runs[1] = RunData(path=tuned.path, meta=tuned.meta, records=records)
    analysis = analyze(runs, settings=SETTINGS)
    html = render_html(analysis)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    markdown = render_markdown(analysis)
    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
    assert "\\| pipe" in markdown


def test_unreliable_judge_is_marked(tmp_path: Path) -> None:
    bad = calibrate(
        [JudgeVerdict.CORRECT, JudgeVerdict.INCORRECT] * 10, [JudgeVerdict.CORRECT] * 20
    )
    analysis = analyze(_runs(tmp_path), settings=SETTINGS, calibration=bad)
    html = render_html(analysis)
    markdown = render_markdown(analysis)
    assert "unreliable judge" in html
    assert "Judge-based results in this report should not be trusted" in html
    assert "⚠ unreliable judge" in markdown
    assert "**unreliable**" in markdown
    assert "kappa = 0.00 < 0.60" in markdown


def test_uncalibrated_judge_gets_a_note(analysis: ExperimentAnalysis) -> None:
    assert "not calibrated against human labels" in render_markdown(analysis)
    assert "not calibrated against human labels" in render_html(analysis)


def test_metadata_lists_runs_and_eval_sets(analysis: ExperimentAnalysis) -> None:
    markdown = render_markdown(analysis)
    assert "## Run metadata" in markdown
    assert "base (baseline)" in markdown
    assert "hash-1" in markdown
    assert "Statistics: 95% percentile bootstrap intervals (500 resamples)" in markdown


def test_charts(analysis: ExperimentAnalysis) -> None:
    chart = headline_chart(analysis)
    assert chart.count('class="mark ') == len(analysis.metrics) * len(analysis.models)
    assert "aria-label=" in chart
    forest = forest_chart("tuned", "accuracy", analysis.slices["accuracy"], analysis.pairwise[0])
    assert forest is not None
    assert "status-critical" in forest
    assert "▼ regression" in forest
    assert forest_chart("tuned", "accuracy", [], None) is None


@pytest.mark.parametrize(
    ("function", "value", "expected"),
    [
        (fmt.percent, 0.7243, "72.4%"),
        (fmt.percent, None, "n/a"),
        (fmt.points, 0.052, "+5.2 pts"),
        (fmt.points, -0.1, "-10.0 pts"),
        (fmt.p_value, 0.0446, "0.045"),
        (fmt.p_value, 8.3e-40, "8.3e-40"),
        (fmt.p_value, 0.0, "< 1e-300"),
        (fmt.mde, 0.041, "±4.1 pts"),
        (fmt.mde, None, "n/a"),
        (fmt.kappa, float("nan"), "undefined"),
        (fmt.timestamp, "2026-10-30T14:57:56+00:00", "2026-10-30 14:57 UTC"),
        (fmt.timestamp, "not a date", "not a date"),
    ],
)
def test_formatting(function: object, value: object, expected: str) -> None:
    assert function(value) == expected  # type: ignore[operator]


def test_interval_and_truncate() -> None:
    assert fmt.interval(0.1, 0.2) == "10.0 to 20.0"
    assert fmt.interval(-0.1, 0.2, signed=True) == "-10.0 to +20.0"
    assert fmt.truncate("x" * 10, 5) == "xxxx…"
