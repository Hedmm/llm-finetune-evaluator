from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

from helpers import write_jsonl
from llm_finetune_evaluator.backends.engine import GenerationEngine
from llm_finetune_evaluator.backends.mock import MockJudgeBackend
from llm_finetune_evaluator.calibration import (
    CalibrationResult,
    HumanLabel,
    calibrate,
    calibrate_judge,
    labeling_template,
    load_calibration,
    load_human_labels,
    save_calibration,
)
from llm_finetune_evaluator.config import MockJudgeConfig, SamplingParams
from llm_finetune_evaluator.dataset import EvalItem, TaskType
from llm_finetune_evaluator.errors import DatasetError
from llm_finetune_evaluator.rubric import JudgeVerdict
from llm_finetune_evaluator.scorers.judge import JudgeScorer
from llm_finetune_evaluator.stats import cohen_kappa

CORRECT = JudgeVerdict.CORRECT
PARTIAL = JudgeVerdict.PARTIALLY_CORRECT
WRONG = JudgeVerdict.INCORRECT
REFUSED = JudgeVerdict.REFUSED
MADE_UP = JudgeVerdict.MADE_UP


def test_perfect_agreement_is_reliable() -> None:
    human = [CORRECT, PARTIAL, WRONG, REFUSED, MADE_UP] * 4
    result = calibrate(human, list(human))
    assert result.agreement == 1.0
    assert result.kappa == pytest.approx(1.0)
    assert result.reliable
    assert result.warning is None


def test_low_agreement_is_flagged_as_unreliable() -> None:
    human = [CORRECT] * 10 + [WRONG] * 10
    judge = [CORRECT] * 7 + [WRONG] * 3 + [CORRECT] * 4 + [WRONG] * 6
    result = calibrate(human, judge)
    assert result.agreement == pytest.approx(0.65)
    assert result.kappa == pytest.approx(0.3)
    assert not result.reliable
    assert result.warning is not None
    assert "kappa = 0.30 < 0.60 on 20 labelled outputs" in result.warning


def test_threshold_is_configurable() -> None:
    human = [CORRECT] * 10 + [WRONG] * 10
    judge = [CORRECT] * 7 + [WRONG] * 3 + [CORRECT] * 4 + [WRONG] * 6
    assert calibrate(human, judge, threshold=0.25).reliable


def test_undefined_kappa_is_inconclusive() -> None:
    result = calibrate([CORRECT] * 6, [CORRECT] * 6)
    assert math.isnan(result.kappa)
    assert not result.reliable
    assert result.warning is not None
    assert "inconclusive" in result.warning


def test_unparseable_judge_replies_count_as_disagreement() -> None:
    human = [CORRECT, CORRECT, WRONG, WRONG]
    result = calibrate(human, [CORRECT, None, WRONG, WRONG])
    assert result.categories[-1] == "unparseable"
    assert result.n_unparseable == 1
    assert result.agreement == 0.75
    assert result.kappa == pytest.approx(
        cohen_kappa(
            ["correct", "correct", "incorrect", "incorrect"],
            ["correct", "unparseable", "incorrect", "incorrect"],
        )
    )


def test_confusion_matrix_rows_are_human_verdicts() -> None:
    result = calibrate([CORRECT, CORRECT, WRONG], [CORRECT, WRONG, WRONG])
    rows = dict(zip(result.categories, result.confusion, strict=True))
    assert rows["correct"][:3] == (1, 0, 1)
    assert rows["incorrect"][:3] == (0, 0, 1)
    assert result.per_category()[0] == {
        "verdict": "correct",
        "human_count": 2,
        "judge_agrees": 1,
        "recall": 0.5,
    }


def test_kappa_interval_is_reproducible_and_contains_the_estimate() -> None:
    human = [CORRECT, PARTIAL, WRONG, REFUSED, MADE_UP] * 10
    judge = [CORRECT, CORRECT, WRONG, REFUSED, WRONG] * 10
    first = calibrate(human, judge, seed=3)
    second = calibrate(human, judge, seed=3)
    assert (first.kappa_low, first.kappa_high) == (second.kappa_low, second.kappa_high)
    assert first.kappa_low <= first.kappa <= first.kappa_high


def test_round_trips_through_json(tmp_path: Path) -> None:
    for result in (
        calibrate([CORRECT, WRONG, CORRECT, REFUSED], [CORRECT, WRONG, WRONG, REFUSED]),
        calibrate([CORRECT] * 3, [CORRECT] * 3),
    ):
        path = tmp_path / "calibration.json"
        save_calibration(result, path)
        loaded = load_calibration(path)
        assert loaded is not None
        assert loaded.to_dict() == result.to_dict()
        assert json.loads(path.read_text())["reliable"] == result.reliable
    assert load_calibration(tmp_path / "missing.json") is None
    assert isinstance(loaded, CalibrationResult)


class TestHumanLabels:
    def test_loads_labels_and_synonyms(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "labels.jsonl",
            [
                {"id": "a", "output": "x", "label": "correct", "model": "tuned", "annotator": "mm"},
                {"id": "b", "output": "y", "label": "Partially Correct"},
                {"id": "c", "output": "z", "label": "made-up"},
            ],
        )
        labels = load_human_labels(path)
        assert [label.label for label in labels] == [CORRECT, PARTIAL, MADE_UP]
        assert labels[0] == HumanLabel("a", "x", CORRECT, model="tuned", annotator="mm")

    def test_reports_invalid_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        path.write_text(
            '{"id": "a", "output": "x", "label": "great"}\n{"id": "b"}\nnot json\n',
            encoding="utf-8",
        )
        with pytest.raises(DatasetError) as caught:
            load_human_labels(path)
        message = str(caught.value)
        assert "line 1: label 'great' is not one of" in message
        assert "line 2: missing output, label" in message
        assert "line 3: invalid JSON" in message

    def test_rejects_missing_and_empty_files(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="not found"):
            load_human_labels(tmp_path / "missing.jsonl")
        (tmp_path / "empty.jsonl").write_text("\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="no human labels"):
            load_human_labels(tmp_path / "empty.jsonl")


def _judge_scorer(noise: float) -> JudgeScorer:
    backend = MockJudgeBackend(
        MockJudgeConfig(name="mock-judge", backend="mock_judge", noise=noise)
    )
    return JudgeScorer("judge_score", engine=GenerationEngine(backend), sampling=SamplingParams())


def test_calibrates_the_judge_against_human_labels() -> None:
    items = {
        f"q{i}": EvalItem(
            id=f"q{i}", input=f"Q{i}?", reference=f"answer {i}", task=TaskType.OPEN_QA
        )
        for i in range(40)
    }
    labels = [HumanLabel(f"q{i}", f"answer {i}", CORRECT) for i in range(20)]
    labels += [HumanLabel(f"q{i}", "something else entirely", WRONG) for i in range(20, 40)]
    exact = asyncio.run(calibrate_judge(_judge_scorer(noise=0.0), labels, items))
    assert exact.agreement == 1.0
    assert exact.kappa == pytest.approx(1.0)
    assert exact.judge == "mock-judge"
    noisy = asyncio.run(calibrate_judge(_judge_scorer(noise=0.5), labels, items))
    assert noisy.kappa < 0.6
    assert not noisy.reliable


def test_calibration_requires_known_items() -> None:
    with pytest.raises(DatasetError, match="not in the eval sets: missing"):
        asyncio.run(calibrate_judge(_judge_scorer(0.0), [HumanLabel("missing", "x", CORRECT)], {}))


def test_labeling_template_is_blind_deduplicated_and_reproducible() -> None:
    first = EvalItem(id="a", input="Q?", reference="yes", task=TaskType.OPEN_QA)
    second = EvalItem(id="b", input="R?", reference="no", task=TaskType.OPEN_QA)
    outputs = [(first, "yes"), (first, "yes"), (first, "maybe"), (second, "no")]
    rows = labeling_template(outputs, size=10, seed=1)
    assert len(rows) == 3
    assert all(set(row) == {"id", "question", "reference", "output", "label"} for row in rows)
    assert all(row["label"] is None for row in rows)
    assert labeling_template(outputs, size=2, seed=1) == labeling_template(outputs, size=2, seed=1)
    assert len(labeling_template(outputs, size=2, seed=1)) == 2
