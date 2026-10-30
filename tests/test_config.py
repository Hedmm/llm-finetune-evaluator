from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_finetune_evaluator.config import (
    ExperimentConfig,
    JudgeScorerConfig,
    MockModelConfig,
    OpenAIModelConfig,
    ReplayModelConfig,
    SamplingOverrides,
    SamplingParams,
    load_config,
    parse_config,
)
from llm_finetune_evaluator.errors import ConfigError


def _config(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "demo",
        "eval_sets": [{"name": "qa", "path": "data/qa.jsonl"}],
        "models": [
            {"name": "base", "backend": "mock", "accuracy": 0.6},
            {"name": "tuned", "backend": "mock", "accuracy": 0.8},
        ],
        "scorers": [{"type": "normalized_match"}, {"type": "refusal"}],
    }
    data.update(overrides)
    return data


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "experiment" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_loads_a_config_and_resolves_paths_against_its_directory(tmp_path: Path) -> None:
    data = _config(
        models=[
            {"name": "base", "backend": "mock"},
            {"name": "recorded", "backend": "replay", "predictions": "preds/tuned.jsonl"},
        ]
    )
    config = load_config(_write(tmp_path, data))
    assert config.eval_sets[0].path == tmp_path / "experiment" / "data" / "qa.jsonl"
    replay = config.models[1]
    assert isinstance(replay, ReplayModelConfig)
    assert replay.predictions == tmp_path / "experiment" / "preds" / "tuned.jsonl"


def test_defaults() -> None:
    config = parse_config(_config())
    assert config.baseline_name == "base"
    assert config.resolved_output_dir() == Path("runs") / "demo"
    assert config.sampling == SamplingParams()
    assert config.stats.alpha == 0.05
    assert config.stats.bootstrap_resamples == 10_000
    assert config.cache.enabled
    assert [scorer.resolved_name for scorer in config.scorers] == [
        "normalized_match",
        "needless_refusal",
    ]


def test_explicit_baseline_and_output_dir(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _config(baseline="tuned", output_dir="out")))
    assert config.baseline_name == "tuned"
    assert config.resolved_output_dir() == tmp_path / "experiment" / "out"


def test_expands_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    models = [
        {
            "name": "base",
            "backend": "openai",
            "model": "m",
            "base_url": "${LFE_TEST_URL:-http://localhost:8000/v1}",
        },
        {"name": "tuned", "backend": "openai", "model": "${LFE_TEST_MODEL}"},
    ]
    monkeypatch.setenv("LFE_TEST_MODEL", "support-lora")
    config = parse_config(_config(models=models))
    base, tuned = config.models
    assert isinstance(base, OpenAIModelConfig)
    assert isinstance(tuned, OpenAIModelConfig)
    assert base.base_url == "http://localhost:8000/v1"
    assert tuned.model == "support-lora"
    monkeypatch.setenv("LFE_TEST_URL", "http://gpu-box:8001/v1")
    assert parse_config(_config(models=models)).models[0].base_url == "http://gpu-box:8001/v1"  # type: ignore[union-attr]


def test_missing_environment_variable_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LFE_TEST_MODEL", raising=False)
    models = [{"name": "base", "backend": "openai", "model": "${LFE_TEST_MODEL}"}]
    with pytest.raises(ConfigError, match="LFE_TEST_MODEL is not set"):
        parse_config(_config(models=models))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"models": [{"name": "a", "backend": "telepathy"}]},
            "does not match any of the expected tags",
        ),
        (
            {"models": [{"name": "a", "backend": "mock"}, {"name": "a", "backend": "mock"}]},
            "repeated: ['a']",
        ),
        ({"baseline": "nope"}, "baseline 'nope' is not one of the models"),
        ({"scorers": [{"type": "judge"}]}, "needs a 'judge' section"),
        ({"models": [{"name": "j", "backend": "mock_judge"}]}, "can only be used as the judge"),
        (
            {"judge": {"model": {"backend": "mock"}}, "scorers": [{"type": "judge"}]},
            "cannot act as a judge",
        ),
        ({"scorers": [{"type": "regex"}]}, "exactly one of pattern"),
        ({"scorers": [{"type": "regex", "pattern": "("}]}, "invalid regular expression"),
        ({"scorers": [{"type": "trap", "eval_sets": ["other"]}]}, "unknown eval sets ['other']"),
        ({"report": {"primary_metric": "accuracy"}}, "report.primary_metric 'accuracy'"),
        ({"sampling": {"temprature": 0.2}}, "Extra inputs are not permitted"),
        ({"models": [{"name": "bad name", "backend": "mock"}]}, "should match pattern"),
        (
            {"models": [{"name": "a", "backend": "mock", "accuracy": 0.8, "partial_rate": 0.3}]},
            "must not exceed 1",
        ),
        (
            {"models": [{"name": "a", "backend": "mock", "by_tag": {"billing": {}}}]},
            "topic=billing",
        ),
        ({"scorers": [{"type": "label_set", "labels": ["a"], "labels_file": "l.txt"}]}, "not both"),
        ({"eval_sets": []}, "at least 1 item"),
    ],
)
def test_reports_invalid_configs_readably(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError) as caught:
        parse_config(_config(**overrides), source="config.yaml")
    assert str(caught.value).startswith("config.yaml is not a valid config:")
    assert message in str(caught.value)


def test_judge_model_name_defaults_to_judge() -> None:
    config = parse_config(
        _config(
            scorers=[{"type": "judge", "partial_credit": 0.25}],
            judge={"model": {"backend": "mock_judge", "noise": 0.2}},
        )
    )
    assert config.judge is not None
    assert config.judge.model.name == "judge"
    assert config.judge.sampling.temperature == 0.0
    scorer = config.scorers[0]
    assert isinstance(scorer, JudgeScorerConfig)
    assert scorer.resolved_name == "judge_score"


def test_sampling_overrides_replace_only_the_fields_they_set() -> None:
    base = SamplingParams(temperature=0.0, max_tokens=128, stop=("\n",))
    merged = SamplingOverrides(max_tokens=16).apply(base)
    assert merged == SamplingParams(temperature=0.0, max_tokens=16, stop=("\n",))


def test_api_key_is_only_sent_to_openai_by_default() -> None:
    hosted = OpenAIModelConfig(name="a", backend="openai", model="gpt-4o-mini")
    local = OpenAIModelConfig(
        name="b", backend="openai", model="m", base_url="http://localhost:11434/v1"
    )
    explicit = OpenAIModelConfig(
        name="c",
        backend="openai",
        model="m",
        base_url="http://localhost:8000/v1",
        api_key_env="VLLM_KEY",
    )
    assert hosted.resolved_api_key_env() == "OPENAI_API_KEY"
    assert local.resolved_api_key_env() is None
    assert explicit.resolved_api_key_env() == "VLLM_KEY"


def test_mock_by_tag_overrides_are_parsed() -> None:
    config = parse_config(
        _config(
            models=[
                {"name": "a", "backend": "mock", "by_tag": {"topic=billing": {"accuracy": 0.2}}}
            ]
        )
    )
    model = config.models[0]
    assert isinstance(model, MockModelConfig)
    assert model.by_tag["topic=billing"].accuracy == 0.2


def test_rejects_missing_invalid_and_non_mapping_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(broken)
    listing = tmp_path / "list.yaml"
    listing.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_config(listing)


def test_config_is_immutable() -> None:
    config = parse_config(_config())
    assert isinstance(config, ExperimentConfig)
    with pytest.raises(ValueError, match="frozen"):
        config.name = "other"  # type: ignore[misc]
