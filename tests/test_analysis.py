from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from helpers import BINARY, make_run
from llm_finetune_evaluator.analysis import MCNEMAR, PERMUTATION, AnalysisSettings, analyze
from llm_finetune_evaluator.calibration import calibrate
from llm_finetune_evaluator.errors import RunError
from llm_finetune_evaluator.rubric import JudgeVerdict
from llm_finetune_evaluator.scorers.base import MetricInfo
from llm_finetune_evaluator.stats import mcnemar_exact

FAST = AnalysisSettings(bootstrap_resamples=1_000, permutation_resamples=1_000, min_slice_size=10)
IDS = [f"q{i:03d}" for i in range(200)]
TOPICS = {item_id: {"topic": "loans" if i % 4 == 0 else "cards"} for i, item_id in enumerate(IDS)}


def _values(seed: int, accuracy: float, loans_accuracy: float | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = {}
    for i, item_id in enumerate(IDS):
        rate = loans_accuracy if loans_accuracy is not None and i % 4 == 0 else accuracy
        values[item_id] = float(rng.random() < rate)
    return values


def test_detects_a_planted_slice_regression(tmp_path: Path) -> None:
    base = make_run(tmp_path, "base", _values(1, 0.7), tags=TOPICS, is_baseline=True)
    tuned = make_run(
        tmp_path, "tuned", _values(2, 0.9, loans_accuracy=0.2), tags=TOPICS, model_index=1
    )
    analysis = analyze([base, tuned], settings=FAST)
    assert analysis.baseline == "base"
    loans = [r for r in analysis.regressions if r.slice_label == "topic=loans"]
    assert len(loans) == 1
    assert loans[0].difference < -0.3
    assert loans[0].family_size == 2
    cards = next(row for row in analysis.slices["accuracy"] if row.value == "cards")
    assert cards.comparisons["tuned"].outcome == "better"


def test_identical_models_have_no_regressions(tmp_path: Path) -> None:
    values = _values(3, 0.7)
    analysis = analyze(
        [
            make_run(tmp_path, "a", values, tags=TOPICS),
            make_run(tmp_path, "b", values, tags=TOPICS),
        ],
        settings=FAST,
    )
    [comparison] = analysis.pairwise
    assert (comparison.difference, comparison.p_value, comparison.outcome) == (
        0.0,
        1.0,
        "no significant difference",
    )
    assert comparison.mde is None
    assert analysis.regressions == []
    assert analysis.possible_regressions == []


def test_pairwise_matches_mcnemar_and_reports_the_counts(tmp_path: Path) -> None:
    base_values, tuned_values = _values(4, 0.6), _values(5, 0.75)
    analysis = analyze(
        [make_run(tmp_path, "base", base_values), make_run(tmp_path, "tuned", tuned_values)],
        settings=FAST,
    )
    [comparison] = analysis.pairwise
    expected = mcnemar_exact(list(tuned_values.values()), list(base_values.values()))
    assert comparison.test == MCNEMAR
    assert comparison.p_value == pytest.approx(expected.p_value)
    assert (comparison.candidate_better, comparison.baseline_better) == (
        expected.a_only,
        expected.b_only,
    )
    assert comparison.difference == pytest.approx(
        np.mean(list(tuned_values.values())) - np.mean(list(base_values.values()))
    )
    assert comparison.ci_low < comparison.difference < comparison.ci_high
    assert comparison.mde is not None
    assert comparison.mde > 0


def test_lower_is_better_metrics_flip_the_direction(tmp_path: Path) -> None:
    refusal = MetricInfo("needless_refusal", "refusal", "Refusals", False, True, ("open_qa",))
    base = make_run(tmp_path, "base", _values(6, 0.02), metrics=[refusal])
    tuned = make_run(tmp_path, "tuned", _values(7, 0.20), metrics=[refusal])
    analysis = analyze([base, tuned], settings=FAST)
    [comparison] = analysis.pairwise
    assert comparison.difference > 0
    assert comparison.improvement < 0
    assert comparison.outcome == "worse"
    assert analysis.regressions == [comparison]


def test_continuous_metrics_use_the_permutation_test(tmp_path: Path) -> None:
    f1 = MetricInfo("field_f1", "json_fields", "F1", True, False, ("extraction",))
    rng = np.random.default_rng(8)
    base = {item_id: float(v) for item_id, v in zip(IDS, rng.random(200), strict=True)}
    tuned = {item_id: min(1.0, v + 0.1) for item_id, v in base.items()}
    analysis = analyze(
        [make_run(tmp_path, "b", base, metrics=[f1]), make_run(tmp_path, "t", tuned, metrics=[f1])],
        settings=FAST,
    )
    [comparison] = analysis.pairwise
    assert comparison.test == PERMUTATION
    assert comparison.outcome == "better"


def test_holm_family_covers_all_model_pairs(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path, name, _values(seed, rate))
        for name, seed, rate in [("a", 9, 0.6), ("b", 10, 0.7), ("c", 11, 0.8)]
    ]
    analysis = analyze(runs, baseline="a", settings=FAST)
    assert [(c.baseline, c.candidate) for c in analysis.pairwise] == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    assert {c.family_size for c in analysis.pairwise} == {3}
    raw = sorted(c.p_value for c in analysis.pairwise)
    assert min(c.p_adjusted for c in analysis.pairwise) == pytest.approx(min(1.0, 3 * raw[0]))
    assert [len(e.wins) <= FAST.max_examples for e in analysis.examples] == [True, True]


def test_unscored_items_are_left_out_of_the_pairing(tmp_path: Path) -> None:
    base = dict.fromkeys(IDS[:50], 1.0)
    tuned: dict[str, float | None] = dict.fromkeys(IDS[:50], 1.0)
    tuned[IDS[0]] = None
    analysis = analyze(
        [make_run(tmp_path, "base", base), make_run(tmp_path, "tuned", tuned)], settings=FAST
    )
    assert analysis.pairwise[0].n == 49
    assert analysis.headline["accuracy"]["tuned"].n_unscored == 1
    estimate = analysis.headline["accuracy"]["tuned"].estimate
    assert estimate is not None
    assert estimate.n == 49


def test_slices_with_identical_items_are_tested_once(tmp_path: Path) -> None:
    tags = {
        item_id: {"topic": t["topic"], "product": "loan" if t["topic"] == "loans" else "card"}
        for item_id, t in TOPICS.items()
    }
    base = make_run(tmp_path, "base", _values(12, 0.7), tags=tags)
    tuned = make_run(tmp_path, "tuned", _values(13, 0.7), tags=tags)
    rows = analyze([base, tuned], settings=FAST).slices["accuracy"]
    assert [(row.dimension, row.value, row.aliases) for row in rows] == [
        ("product", "card", ("topic=cards",)),
        ("product", "loan", ("topic=loans",)),
    ]
    assert {row.comparisons["tuned"].family_size for row in rows} == {2}


def test_small_slices_are_not_reported(tmp_path: Path) -> None:
    tags = {item_id: {"region": "rare" if i < 5 else "common"} for i, item_id in enumerate(IDS)}
    rows = analyze(
        [
            make_run(tmp_path, "a", _values(14, 0.5), tags=tags),
            make_run(tmp_path, "b", _values(15, 0.5), tags=tags),
        ],
        settings=FAST,
    ).slices["accuracy"]
    assert [row.value for row in rows] == ["common"]


def test_runs_on_different_eval_sets_cannot_be_compared(tmp_path: Path) -> None:
    a = make_run(tmp_path, "a", _values(16, 0.5), eval_set_hash="one")
    b = make_run(tmp_path, "b", _values(17, 0.5), eval_set_hash="two")
    with pytest.raises(RunError, match="different eval set contents"):
        analyze([a, b])


def test_duplicate_model_names_are_rejected(tmp_path: Path) -> None:
    a = make_run(tmp_path / "x", "a", _values(18, 0.5))
    b = make_run(tmp_path / "y", "a", _values(19, 0.5))
    with pytest.raises(RunError, match="same model name"):
        analyze([a, b])


def test_baseline_resolution(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path, "first", _values(20, 0.5)),
        make_run(tmp_path, "flagged", _values(21, 0.5), is_baseline=True),
        make_run(tmp_path, "third", _values(22, 0.5)),
    ]
    assert analyze(runs, settings=FAST).baseline == "flagged"
    assert analyze(runs, settings=FAST, experiment={"baseline": "third"}).baseline == "third"
    assert analyze(runs, baseline="first", settings=FAST).baseline == "first"
    assert analyze(runs, baseline="first", settings=FAST).models[0] == "first"
    with pytest.raises(RunError, match="baseline 'nope'"):
        analyze(runs, baseline="nope")


def test_examples_show_the_largest_differences(tmp_path: Path) -> None:
    f1 = MetricInfo("field_f1", "json_fields", "F1", True, False, ("extraction",))
    base = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
    tuned = {"a": 1.0, "b": 0.6, "c": 0.5, "d": 0.0}
    analysis = analyze(
        [
            make_run(tmp_path, "base", base, metrics=[f1]),
            make_run(tmp_path, "tuned", tuned, metrics=[f1]),
        ],
        settings=AnalysisSettings(
            bootstrap_resamples=200, permutation_resamples=200, max_examples=1
        ),
    )
    [examples] = analysis.examples
    assert (examples.n_wins, examples.n_losses) == (2, 1)
    assert [row.id for row in examples.wins] == ["a"]
    assert [row.id for row in examples.losses] == ["d"]
    assert examples.wins[0].candidate_output == "tuned output for a"


def test_judge_status_follows_the_calibration(tmp_path: Path) -> None:
    judge = MetricInfo("judge_score", "judge", "Judge", True, False, ("open_qa",), uses_judge=True)
    runs = [
        make_run(tmp_path, "a", _values(23, 0.5), metrics=[judge]),
        make_run(tmp_path, "b", _values(24, 0.6), metrics=[judge]),
    ]
    assert analyze(runs, settings=FAST).judge_status == "uncalibrated"
    good = calibrate(
        [JudgeVerdict.CORRECT, JudgeVerdict.INCORRECT] * 10,
        [JudgeVerdict.CORRECT, JudgeVerdict.INCORRECT] * 10,
    )
    bad = calibrate(
        [JudgeVerdict.CORRECT, JudgeVerdict.INCORRECT] * 10, [JudgeVerdict.CORRECT] * 20
    )
    assert analyze(runs, settings=FAST, calibration=good).judge_status == "reliable"
    unreliable = analyze(runs, settings=FAST, calibration=bad)
    assert unreliable.judge_status == "unreliable"
    assert unreliable.judge_unreliable(judge)
    assert any("kappa" in warning for warning in unreliable.warnings)
    assert (
        analyze([make_run(tmp_path / "plain", "a", _values(25, 0.5))], settings=FAST).judge_status
        == "not_used"
    )


def test_extraction_summary_is_micro_averaged(tmp_path: Path) -> None:
    f1 = MetricInfo("field_f1", "json_fields", "F1", True, False, ("extraction",))
    details = {"a": {"tp": 3, "fp": 1, "fn": 0}, "b": {"tp": 1, "fp": 0, "fn": 3}}
    run = make_run(tmp_path, "m", {"a": 0.86, "b": 0.4}, metrics=[f1], details=details)
    summary = analyze([run], settings=FAST).extraction["field_f1"]["m"]
    assert summary["precision"] == pytest.approx(4 / 5)
    assert summary["recall"] == pytest.approx(4 / 7)


def test_settings_come_from_the_experiment_with_overrides() -> None:
    experiment = {
        "stats": {"alpha": 0.01, "seed": 7, "slice_by": ["topic"]},
        "report": {"max_examples": 2},
    }
    settings = AnalysisSettings.from_experiment(experiment, seed=3, alpha=None)
    assert (settings.alpha, settings.seed, settings.slice_by, settings.max_examples) == (
        0.01,
        3,
        ("topic",),
        2,
    )


def test_slice_by_restricts_the_dimensions(tmp_path: Path) -> None:
    tags = {item_id: {"topic": t["topic"], "length": "long"} for item_id, t in TOPICS.items()}
    runs = [
        make_run(tmp_path, "a", _values(26, 0.5), tags=tags),
        make_run(tmp_path, "b", _values(27, 0.5), tags=tags),
    ]
    analysis = analyze(
        runs, settings=AnalysisSettings(bootstrap_resamples=200, slice_by=("topic", "region"))
    )
    assert analysis.slice_dimensions == ["topic"]
    assert any("'region'" in warning for warning in analysis.warnings)


def test_analysis_is_reproducible_and_strict_json(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path, "a", _values(28, 0.5), tags=TOPICS),
        make_run(tmp_path, "b", _values(29, 0.7), tags=TOPICS),
    ]
    first = analyze(runs, settings=FAST).to_dict()
    second = analyze(runs, settings=FAST).to_dict()
    for data in (first, second):
        data.pop("created_at")
    assert first == second

    def reject(value: str) -> None:
        raise ValueError(value)

    text = json.dumps(first, allow_nan=False)
    assert json.loads(text, parse_constant=reject)["models"] == ["a", "b"]
    assert not any(
        isinstance(v, float) and math.isnan(v)
        for v in first["pairwise"][0].values()
        if isinstance(v, float)
    )


def test_single_run_gives_headline_only(tmp_path: Path) -> None:
    analysis = analyze([make_run(tmp_path, "only", _values(30, 0.5), tags=TOPICS)], settings=FAST)
    assert analysis.pairwise == []
    assert analysis.candidates == []
    assert analysis.examples == []
    assert analysis.headline["accuracy"]["only"].estimate is not None
    assert BINARY.name in analysis.slices
