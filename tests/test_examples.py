"""The examples are part of the product: they must stay valid and reproducible."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from helpers import read_jsonl
from llm_finetune_evaluator.analysis import AnalysisSettings, analyze
from llm_finetune_evaluator.cli import app
from llm_finetune_evaluator.config import load_config
from llm_finetune_evaluator.dataset import load_eval_set
from llm_finetune_evaluator.runner import run_experiment
from llm_finetune_evaluator.scorers import ExactMatchScorer
from llm_finetune_evaluator.storage import load_run

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
CONFIGS = sorted(EXAMPLES.glob("*/config*.yaml"))
BANKING = EXAMPLES / "banking77"


def _load_module(path: Path) -> ModuleType:
    name = f"example_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_every_example_has_a_config() -> None:
    assert {path.parent.name for path in CONFIGS} == {
        "banking77",
        "mock_quickstart",
        "trap_questions",
    }


@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_example_configs_pass_a_dry_run(config: Path) -> None:
    result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["run", str(config), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "is valid" in result.output


class TestBanking77:
    def test_slice(self) -> None:
        eval_set = load_eval_set(BANKING / "data" / "banking77_slice.jsonl")
        labels = (BANKING / "labels.txt").read_text(encoding="utf-8").split()
        assert len(eval_set.items) == 300
        assert len(labels) == 77
        assert {item.reference for item in eval_set.items} <= set(labels)
        assert len({item.reference for item in eval_set.items}) == 76
        assert {tuple(sorted(item.tags)) for item in eval_set.items} == {("group", "length")}

    def test_groups_cover_every_intent(self) -> None:
        module = _load_module(BANKING / "make_slice.py")
        labels = (BANKING / "labels.txt").read_text(encoding="utf-8").split()
        assert sorted(module.GROUP_OF) == sorted(labels)
        assert sum(len(intents) for intents in module.GROUPS.values()) == 77

    @pytest.mark.parametrize("arm", ["base", "lora", "qlora", "full"])
    def test_exact_match_agrees_with_the_case_study(self, arm: str) -> None:
        items = {
            item.id: item
            for item in load_eval_set(BANKING / "data" / "banking77_slice.jsonl").items
        }
        scorer = ExactMatchScorer("accuracy")
        rows = read_jsonl(BANKING / "predictions" / f"{arm}.jsonl")
        assert len(rows) == 300
        for row in rows:
            assert scorer.score(items[row["id"]], row["output"]).value == float(
                row["case_study_correct"]
            )

    def test_readme_numbers(self, tmp_path: Path) -> None:
        result = run_experiment(load_config(BANKING / "config_replay.yaml"), output_dir=tmp_path)
        runs = [load_run(result.runs[arm]) for arm in ("base", "lora", "qlora", "full")]
        analysis = analyze(runs, baseline="base", settings=AnalysisSettings())
        accuracy = {m: analysis.headline["accuracy"][m].estimate for m in analysis.models}
        lenient = {m: analysis.headline["label_accuracy"][m].estimate for m in analysis.models}
        expected = {"base": 113, "lora": 278, "qlora": 278, "full": 258}
        for model, correct in expected.items():
            estimate = accuracy[model]
            assert estimate is not None
            assert estimate.value == pytest.approx(correct / 300)
        base_lenient = lenient["base"]
        assert base_lenient is not None
        assert base_lenient.value == pytest.approx(130 / 300)
        base_accuracy = accuracy["base"]
        assert base_accuracy is not None
        assert (round(100 * base_accuracy.low, 1), round(100 * base_accuracy.high, 1)) == (
            32.3,
            43.3,
        )
        pairs = {(c.baseline, c.candidate): c for c in analysis.pairwise if c.metric == "accuracy"}
        lora = pairs[("base", "lora")]
        assert (lora.candidate_better, lora.baseline_better) == (166, 1)
        assert f"{lora.p_adjusted:.1e}" == "1.1e-47"
        full = pairs[("lora", "full")]
        assert (full.candidate_better, full.baseline_better) == (4, 24)
        assert f"{full.p_value:.1e}" == "1.8e-04"
        assert f"{full.p_adjusted:.1e}" == "5.4e-04"
        assert (round(100 * full.ci_low, 1), round(100 * full.ci_high, 1)) == (-10.0, -3.3)
        assert full.mde == pytest.approx(0.049, abs=5e-4)
        qlora = pairs[("lora", "qlora")]
        assert (qlora.candidate_better, qlora.baseline_better, qlora.p_value) == (5, 5, 1.0)
        assert qlora.mde == pytest.approx(0.029, abs=5e-4)


def test_trap_questions_set() -> None:
    eval_set = load_eval_set(EXAMPLES / "trap_questions" / "data.jsonl")
    counts = Counter(item.task.value for item in eval_set.items)
    assert counts == {"open_qa": 24, "trap": 12, "refusal_expected": 4}
    for item in eval_set.items:
        if item.task.value != "refusal_expected":
            assert isinstance(item.input, str)
            assert item.input.startswith("Source:\n")


def test_mock_quickstart_data_matches_its_generator() -> None:
    folder = EXAMPLES / "mock_quickstart"
    module = _load_module(folder / "make_data.py")
    rng = random.Random(module.SEED)
    support = module.support_items(rng)
    assert support == read_jsonl(folder / "data" / "support_qa.jsonl")
    assert module.intent_items(rng) == read_jsonl(folder / "data" / "intents.jsonl")
    assert module.extraction_items(rng) == read_jsonl(folder / "data" / "extraction.jsonl")
    labels = module.human_labels(support, 80, rng)
    assert labels == read_jsonl(folder / "data" / "human_labels.jsonl")
    assert json.dumps(labels)
