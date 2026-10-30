from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from helpers import mock_experiment, qa_rows, read_jsonl, write_jsonl
from llm_finetune_evaluator.config import ExperimentConfig, load_config, parse_config
from llm_finetune_evaluator.dataset import ChatMessage, TaskType
from llm_finetune_evaluator.errors import ConfigError, DatasetError, RunError
from llm_finetune_evaluator.runner import judge_family_warnings, run_experiment
from llm_finetune_evaluator.scorers.base import Score
from llm_finetune_evaluator.storage import (
    EXPERIMENT_FILE,
    ItemRecord,
    discover_runs,
    load_experiment,
    load_run,
    write_run,
)


def test_run_writes_one_directory_per_model(tmp_path: Path) -> None:
    config_path = mock_experiment(tmp_path)
    result = run_experiment(load_config(config_path), config_path=config_path)
    assert result.output_dir == tmp_path / "runs"
    assert set(result.runs) == {"base", "tuned"}
    run = load_run(result.runs["tuned"])
    assert len(run.records) == 120
    assert run.meta["is_baseline"] is False
    assert run.meta["generation"] == {"requested": 120, "cached": 0, "generated": 120, "retries": 0}
    assert [metric.name for metric in run.metrics] == ["normalized_match", "needless_refusal"]
    record = run.records[0]
    assert record.messages == (ChatMessage(role="user", content="Question 0?"),)
    assert set(record.scores) == {"normalized_match", "needless_refusal"}
    experiment = load_experiment(result.output_dir)
    assert experiment is not None
    assert experiment["baseline"] == "base"
    assert experiment["models"] == ["base", "tuned"]
    assert experiment["eval_sets"][0]["n_items"] == 120


def test_rerun_is_served_from_the_cache(tmp_path: Path) -> None:
    config_path = mock_experiment(tmp_path)
    config = load_config(config_path)
    run_experiment(config)
    again = run_experiment(config)
    meta = load_run(again.runs["base"]).meta
    assert meta["generation"]["cached"] == 120
    assert meta["generation"]["generated"] == 0
    first = [r.output for r in load_run(again.runs["base"]).records]
    no_cache = run_experiment(config, use_cache=False)
    assert [r.output for r in load_run(no_cache.runs["base"]).records] == first


def test_selected_models_only_replace_their_own_runs(tmp_path: Path) -> None:
    config = load_config(mock_experiment(tmp_path))
    run_experiment(config)
    base_before = (tmp_path / "runs" / "base" / "run.json").read_text()
    result = run_experiment(config, models=["tuned"])
    assert set(result.runs) == {"tuned"}
    assert (tmp_path / "runs" / "base" / "run.json").read_text() == base_before
    with pytest.raises(ConfigError, match="unknown models"):
        run_experiment(config, models=["nope"])


def test_reasoning_blocks_are_removed_before_scoring(tmp_path: Path) -> None:
    rows = qa_rows(2)
    models: list[dict[str, Any]] = [
        {"name": "base", "backend": "mock", "responses": {"q000": "<think>hmm</think>answer 0"}},
        {
            "name": "raw",
            "backend": "mock",
            "strip_think_tags": False,
            "responses": {"q000": "<think>hmm</think>answer 0"},
        },
    ]
    result = run_experiment(load_config(mock_experiment(tmp_path, rows=rows, models=models)))
    cleaned = load_run(result.runs["base"]).records[0]
    raw = load_run(result.runs["raw"]).records[0]
    assert (cleaned.output, cleaned.raw_output) == ("answer 0", "<think>hmm</think>answer 0")
    assert cleaned.scores["normalized_match"].value == 1.0
    assert raw.raw_output is None
    assert raw.scores["normalized_match"].value == 0.0


def test_prompt_errors_surface_before_any_model_call(tmp_path: Path) -> None:
    config_path = mock_experiment(
        tmp_path,
        extra={
            "eval_sets": [
                {"name": "qa", "path": "data/qa.jsonl", "prompt": {"template": "{{ nope }}"}}
            ]
        },
    )
    with pytest.raises(DatasetError, match="'nope' is undefined"):
        run_experiment(load_config(config_path))
    assert not (tmp_path / "runs").exists()


def test_ids_must_be_unique_across_eval_sets(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "data" / "other.jsonl", qa_rows(3))
    config_path = mock_experiment(
        tmp_path,
        extra={
            "eval_sets": [
                {"name": "qa", "path": "data/qa.jsonl"},
                {"name": "other", "path": "data/other.jsonl"},
            ]
        },
    )
    with pytest.raises(DatasetError, match="occurs in eval sets 'qa' and 'other'"):
        run_experiment(load_config(config_path))


def test_scorers_without_items_are_skipped_with_a_warning(tmp_path: Path) -> None:
    config_path = mock_experiment(
        tmp_path, scorers=[{"type": "normalized_match"}, {"type": "trap"}]
    )
    result = run_experiment(load_config(config_path))
    assert any("'trap_accuracy' applies to no item" in warning for warning in result.warnings)
    assert [m.name for m in load_run(result.runs["base"]).metrics] == ["normalized_match"]


def test_judge_scores_and_calibration_are_written(tmp_path: Path) -> None:
    labels = write_jsonl(
        tmp_path / "labels.jsonl",
        [{"id": f"q{i:03d}", "output": f"answer {i}", "label": "correct"} for i in range(10)]
        + [{"id": f"q{i:03d}", "output": "wrong", "label": "incorrect"} for i in range(10, 20)],
    )
    config_path = mock_experiment(
        tmp_path,
        scorers=[{"type": "normalized_match"}, {"type": "judge"}],
        extra={
            "judge": {
                "model": {"backend": "mock_judge", "noise": 0.0},
                "calibration": {"labels": str(labels)},
            }
        },
    )
    result = run_experiment(load_config(config_path))
    assert result.calibration is not None
    assert result.calibration.kappa == pytest.approx(1.0)
    saved = json.loads((tmp_path / "runs" / "calibration.json").read_text())
    assert saved["reliable"] is True
    scores = [r.scores["judge_score"] for r in load_run(result.runs["tuned"]).records]
    assert {score.label for score in scores} <= {
        "correct",
        "partially_correct",
        "incorrect",
        "refused",
    }
    assert load_run(result.runs["tuned"]).meta["judge"]["kind"] == "mock_judge"


def test_unreliable_judge_calibration_is_reported_as_a_warning(tmp_path: Path) -> None:
    labels = write_jsonl(
        tmp_path / "labels.jsonl",
        [{"id": f"q{i:03d}", "output": f"answer {i}", "label": "incorrect"} for i in range(10)]
        + [{"id": f"q{i:03d}", "output": "wrong", "label": "correct"} for i in range(10, 20)],
    )
    config_path = mock_experiment(
        tmp_path,
        scorers=[{"type": "judge"}],
        extra={
            "judge": {
                "model": {"backend": "mock_judge", "noise": 0.0},
                "calibration": {"labels": str(labels)},
            }
        },
    )
    result = run_experiment(load_config(config_path))
    assert result.calibration is not None
    assert not result.calibration.reliable
    assert any("Cohen's kappa" in warning for warning in result.warnings)


def test_judge_family_warnings() -> None:
    def config(judge_model: str, model: str) -> ExperimentConfig:
        return parse_config(
            {
                "name": "demo",
                "eval_sets": [{"name": "qa", "path": "qa.jsonl"}],
                "models": [
                    {
                        "name": "tuned",
                        "backend": "openai",
                        "model": model,
                        "base_url": "http://localhost:8000/v1",
                    }
                ],
                "scorers": [{"type": "judge"}],
                "judge": {
                    "model": {
                        "backend": "openai",
                        "model": judge_model,
                        "base_url": "http://localhost:8001/v1",
                    }
                },
            }
        )

    same = judge_family_warnings(config("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"))
    family = judge_family_warnings(config("qwen2.5:14b", "Qwen/Qwen2.5-1.5B-Instruct"))
    other = judge_family_warnings(config("llama3.1:8b", "Qwen/Qwen2.5-1.5B-Instruct"))
    assert "same model" in same[0]
    assert "same model family (qwen)" in family[0]
    assert other == []


class TestStorage:
    def test_records_round_trip(self, tmp_path: Path) -> None:
        record = ItemRecord(
            id="a",
            eval_set="qa",
            task=TaskType.TRAP,
            tags={"topic": "x"},
            input="Q?",
            messages=(ChatMessage(role="user", content="Q?"),),
            reference=None,
            output="I don't know.",
            raw_output="<think>.</think>I don't know.",
            finish_reason="stop",
            latency_s=0.5,
            cached=True,
            scores={"trap_accuracy": Score(1.0, "abstained", {"phrase": "I don't know"})},
        )
        write_run(tmp_path / "run", {"model": "m", "eval_set_hash": "h", "metrics": []}, [record])
        loaded = load_run(tmp_path / "run")
        assert loaded.records == (record,)
        assert loaded.model == "m"
        assert not list((tmp_path / "run").glob(".*.tmp"))

    def test_load_run_rejects_incomplete_or_foreign_directories(self, tmp_path: Path) -> None:
        with pytest.raises(RunError, match="not a run directory"):
            load_run(tmp_path)
        write_run(tmp_path / "run", {"model": "m", "eval_set_hash": "h"}, [])
        run_file = tmp_path / "run" / "run.json"
        run_file.write_text(json.dumps({"format_version": 99, "model": "m"}), encoding="utf-8")
        with pytest.raises(RunError, match="run format 99"):
            load_run(tmp_path / "run")
        run_file.write_text("{broken", encoding="utf-8")
        with pytest.raises(RunError, match="corrupt"):
            load_run(tmp_path / "run")

    def test_discover_runs(self, tmp_path: Path) -> None:
        for name in ("b", "a"):
            write_run(tmp_path / "exp" / name, {"model": name, "eval_set_hash": "h"}, [])
        assert discover_runs(tmp_path / "exp") == [tmp_path / "exp" / "a", tmp_path / "exp" / "b"]
        assert discover_runs(tmp_path) == [tmp_path / "exp" / "a", tmp_path / "exp" / "b"]
        assert discover_runs(tmp_path / "exp" / "a") == [tmp_path / "exp" / "a"]
        (tmp_path / "exp" / EXPERIMENT_FILE).write_text(json.dumps({"models": ["b", "a", "gone"]}))
        assert discover_runs(tmp_path / "exp") == [tmp_path / "exp" / "b", tmp_path / "exp" / "a"]
        (tmp_path / "empty").mkdir()
        with pytest.raises(RunError, match="no runs found"):
            discover_runs(tmp_path / "empty")
        with pytest.raises(RunError, match="does not exist"):
            discover_runs(tmp_path / "missing")

    def test_written_records_are_json_lines(self, tmp_path: Path) -> None:
        config_path = mock_experiment(tmp_path, n=4)
        result = run_experiment(load_config(config_path))
        rows = read_jsonl(result.runs["base"] / "records.jsonl")
        assert [row["id"] for row in rows] == ["q000", "q001", "q002", "q003"]
        assert rows[0]["scores"]["normalized_match"]["value"] in (0.0, 1.0)
