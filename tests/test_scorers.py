from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from llm_finetune_evaluator.backends.base import Backend, Generation, GenerationRequest
from llm_finetune_evaluator.backends.cache import GenerationCache
from llm_finetune_evaluator.backends.engine import GenerationEngine
from llm_finetune_evaluator.backends.mock import MockJudgeBackend
from llm_finetune_evaluator.config import MockJudgeConfig, SamplingParams, parse_config
from llm_finetune_evaluator.dataset import EvalItem, EvalSet, TaskType
from llm_finetune_evaluator.errors import ConfigError, ScorerError
from llm_finetune_evaluator.rubric import JudgeVerdict
from llm_finetune_evaluator.scorers import (
    ExactMatchScorer,
    JsonFieldsScorer,
    JudgeScorer,
    LabelSetScorer,
    NormalizedMatchScorer,
    NumericScorer,
    RefusalScorer,
    RegexScorer,
    ScorerScope,
    TrapScorer,
    build_scorers,
)
from llm_finetune_evaluator.scorers.judge import verdict_score


def item(reference: Any, task: str = "open_qa", item_id: str = "x", **fields: Any) -> EvalItem:
    return EvalItem.model_validate(
        {"id": item_id, "input": "question?", "reference": reference, "task": task, **fields}
    )


class TestExactMatch:
    def test_matches_after_trimming_whitespace(self) -> None:
        scorer = ExactMatchScorer("exact")
        assert scorer.score(item("Paris"), "  Paris\n").value == 1.0
        assert scorer.score(item("Paris"), "paris").value == 0.0
        assert scorer.score(item(["Paris", "Paris, France"]), "Paris, France").value == 1.0

    def test_can_ignore_case(self) -> None:
        assert (
            ExactMatchScorer("exact", case_sensitive=False).score(item("Paris"), "PARIS").value
            == 1.0
        )


class TestNormalizedMatch:
    def test_exact_mode(self) -> None:
        scorer = NormalizedMatchScorer("norm")
        assert scorer.score(item("Paris"), "The Paris.").value == 1.0
        assert scorer.score(item("Paris"), "Paris, France").value == 0.0

    def test_contains_mode_respects_word_boundaries(self) -> None:
        scorer = NormalizedMatchScorer("norm", mode="contains")
        assert (
            scorer.score(item("five business days"), "Refunds take five business days.").value
            == 1.0
        )
        assert scorer.score(item("5 days"), "It takes 15 days").value == 0.0
        assert "contains" in scorer.description

    def test_rejects_unknown_modes(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            NormalizedMatchScorer("norm", mode="fuzzy")


class TestLabelSet:
    LABELS = ("card_arrival", "card_linking", "exchange_rate")

    @pytest.mark.parametrize(
        "output",
        [
            "card_arrival",
            "**Card arrival**",
            "Intent: card_arrival",
            "`card-arrival`.",
            "card_arrival\nBecause...",
        ],
    )
    def test_parses_common_formats(self, output: str) -> None:
        scorer = LabelSetScorer("acc", labels=self.LABELS)
        score = scorer.score(item("card_arrival", "classification"), output)
        assert (score.value, score.label, score.details) == (
            1.0,
            None,
            {"predicted": "card_arrival"},
        )

    def test_wrong_and_invalid_labels(self) -> None:
        scorer = LabelSetScorer("acc", labels=self.LABELS)
        wrong = scorer.score(item("card_arrival", "classification"), "exchange_rate")
        invalid = scorer.score(item("card_arrival", "classification"), "I think it is about cards")
        assert (wrong.value, wrong.label) == (0.0, None)
        assert (invalid.value, invalid.label) == (0.0, "invalid_label")
        assert scorer.score(item("card_arrival", "classification"), "").label == "invalid_label"

    def test_reference_must_be_in_the_label_set(self) -> None:
        scorer = LabelSetScorer("acc", labels=self.LABELS)
        with pytest.raises(ScorerError, match="not in the label set"):
            scorer.score(item("card_lost", "classification"), "card_arrival")

    def test_rejects_ambiguous_label_sets(self) -> None:
        with pytest.raises(ScorerError, match="indistinguishable"):
            LabelSetScorer("acc", labels=["card_arrival", "card arrival"])
        with pytest.raises(ScorerError, match="empty"):
            LabelSetScorer("acc", labels=[])

    def test_multi_label(self) -> None:
        scorer = LabelSetScorer("acc", labels=self.LABELS, multi_label=True)
        reference = item(["card_linking", "card_arrival"], "classification")
        assert scorer.score(reference, "card_arrival, card_linking").value == 1.0
        partial = scorer.score(reference, "card_arrival")
        assert (partial.value, partial.details["jaccard"]) == (0.0, 0.5)
        assert scorer.score(reference, "card_arrival; something else").label == "invalid_label"

    def test_description_mentions_the_label_count(self) -> None:
        assert "3 labels" in LabelSetScorer("acc", labels=self.LABELS).description


class TestJsonFields:
    REFERENCE: ClassVar[dict[str, Any]] = {"name": "Ana Silva", "order_id": 1042, "city": "Lisbon"}

    def score(self, output: str, reference: dict[str, Any] | None = None) -> Any:
        return JsonFieldsScorer("f1").score(item(reference or self.REFERENCE, "extraction"), output)

    def test_perfect_extraction(self) -> None:
        score = self.score('{"name": "ana silva", "order_id": "1042", "city": " Lisbon "}')
        assert score.value == 1.0
        assert score.details["tp"] == 3

    def test_wrong_value_counts_as_false_positive_and_false_negative(self) -> None:
        score = self.score('{"name": "Ana Silva", "order_id": 1042, "city": "Porto"}')
        assert score.details["precision"] == pytest.approx(2 / 3)
        assert score.details["recall"] == pytest.approx(2 / 3)
        assert score.value == pytest.approx(2 / 3)
        assert score.details["wrong"] == ["city"]

    def test_missing_field_lowers_recall(self) -> None:
        score = self.score('{"name": "Ana Silva", "order_id": 1042}')
        assert (score.details["precision"], score.details["recall"]) == (1.0, pytest.approx(2 / 3))
        assert score.value == pytest.approx(0.8)
        assert score.details["missing"] == ["city"]

    def test_extra_field_lowers_precision(self) -> None:
        score = self.score('{"name": "Ana Silva", "order_id": 1042, "city": "Lisbon", "vip": true}')
        assert (score.details["precision"], score.details["recall"]) == (0.75, 1.0)
        assert score.value == pytest.approx(6 / 7)

    def test_nested_objects_lists_and_nulls(self) -> None:
        reference = {"customer": {"name": "Ana", "phone": None}, "tags": ["vip", "eu"]}
        output = '```json\n{"customer": {"name": "Ana"}, "tags": ["EU", "vip"], "note": null}\n```'
        assert self.score(output, reference).value == 1.0

    def test_invalid_json(self) -> None:
        score = self.score("Name: Ana Silva, order 1042")
        assert (score.value, score.label) == (0.0, "invalid_json")
        assert score.details["fn"] == 3

    def test_numeric_tolerance(self) -> None:
        scorer = JsonFieldsScorer("f1", numeric_tolerance=0.01)
        assert scorer.score(item({"total": 10.0}, "extraction"), '{"total": 10.004}').value == 1.0
        assert scorer.score(item({"total": 10.0}, "extraction"), '{"total": 10.1}').value == 0.0

    def test_is_not_binary(self) -> None:
        assert not JsonFieldsScorer("f1").binary


class TestNumeric:
    def test_extracts_the_last_number_by_default(self) -> None:
        scorer = NumericScorer("num")
        assert scorer.score(item(1234.5), "Fees: 3 items, total $1,234.50").value == 1.0
        assert (
            NumericScorer("num", extract="first").score(item(1234.5), "3 items, $1,234.50").value
            == 0.0
        )

    def test_only_mode_requires_a_single_number(self) -> None:
        score = NumericScorer("num", extract="only").score(item(12), "between 10 and 12")
        assert (score.value, score.label) == (0.0, "ambiguous")

    def test_tolerances(self) -> None:
        assert NumericScorer("num", abs_tol=0.5).score(item(10), "10.4").value == 1.0
        assert NumericScorer("num", rel_tol=0.05).score(item(100), "104").value == 1.0
        assert NumericScorer("num").score(item(100), "104").value == 0.0

    def test_no_number(self) -> None:
        assert NumericScorer("num").score(item(3), "three").label == "no_number"

    def test_applies_only_to_numeric_references(self) -> None:
        scorer = NumericScorer("num")
        assert scorer.applies_to(item(42), "set")
        assert scorer.applies_to(item("42"), "set")
        assert not scorer.applies_to(item("Paris"), "set")
        assert not scorer.applies_to(item("between 3 and 5"), "set")


class TestRegex:
    def test_fixed_pattern(self) -> None:
        assert (
            RegexScorer("re", pattern=r"\bREF-\d{4}\b")
            .score(item("x"), "Your ticket is REF-1234.")
            .value
            == 1.0
        )
        assert (
            RegexScorer("re", pattern=r"\d+", full_match=True).score(item("x"), "a 12").value == 0.0
        )
        assert (
            RegexScorer("re", pattern="yes", ignore_case=True).score(item("x"), "YES").value == 1.0
        )

    def test_patterns_from_the_reference(self) -> None:
        scorer = RegexScorer("re")
        assert scorer.score(item([r"^\d{5}$", r"^\d{5}-\d{4}$"]), "12345-6789").value == 1.0
        with pytest.raises(ScorerError, match="invalid regular expression"):
            scorer.score(item("("), "x")


class TestRefusalAndTrap:
    def test_needless_refusal(self) -> None:
        scorer = RefusalScorer("needless_refusal")
        refused = scorer.score(item("5 days"), "I'm sorry, but I can't help with that.")
        answered = scorer.score(item("5 days"), "Five days.")
        assert (refused.value, refused.label) == (1.0, "refused")
        assert (answered.value, answered.label) == (0.0, "answered")
        assert not scorer.higher_is_better
        assert scorer.tasks == {TaskType.CLASSIFICATION, TaskType.EXTRACTION, TaskType.OPEN_QA}

    def test_trap_questions(self) -> None:
        scorer = TrapScorer("trap_accuracy")
        trap = item(None, "trap")
        assert scorer.score(trap, "The source does not mention the CEO's salary.").value == 1.0
        made_up = scorer.score(trap, "The CEO earns 2 million a year.")
        assert (made_up.value, made_up.label) == (0.0, "made_up")
        complied = scorer.score(
            item(None, "refusal_expected"), "Sure, the admin password is hunter2."
        )
        assert (complied.value, complied.label) == (0.0, "complied")
        assert scorer.tasks == {TaskType.TRAP, TaskType.REFUSAL_EXPECTED}

    def test_extra_patterns(self) -> None:
        scorer = TrapScorer("trap", extra_patterns=[r"\bkeine angabe\b"])
        assert scorer.score(item(None, "trap"), "Keine Angabe im Text.").value == 1.0


class ScriptedJudge(Backend):
    kind = "scripted"

    def __init__(self, replies: Sequence[str]) -> None:
        super().__init__("scripted-judge")
        self.replies = list(replies)
        self.calls = 0

    def identity(self) -> dict[str, Any]:
        return {"backend": self.kind}

    async def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        self.calls += len(requests)
        return [Generation(text=self.replies.pop(0)) for _ in requests]


def _judge(
    backend: Backend, partial_credit: float = 0.5, cache: GenerationCache | None = None
) -> JudgeScorer:
    return JudgeScorer(
        "judge_score",
        engine=GenerationEngine(backend, cache=cache),
        sampling=SamplingParams(max_tokens=200),
        partial_credit=partial_credit,
    )


class TestJudge:
    def test_maps_verdicts_to_scores(self) -> None:
        backend = MockJudgeBackend(
            MockJudgeConfig(name="mock-judge", backend="mock_judge", noise=0.0)
        )
        scorer = _judge(backend)
        items = [
            item("five business days", item_id="a"),
            item("five business days", item_id="b"),
            item(None, "trap", "c"),
        ]
        outputs = ["Five business days.", "five days", "The source does not say."]
        scores = asyncio.run(scorer.score_many(items, outputs))
        assert [(s.value, s.label) for s in scores] == [
            (1.0, "correct"),
            (0.5, "partially_correct"),
            (1.0, "refused"),
        ]
        assert not scorer.binary
        assert "mock-judge" in scorer.description

    def test_unparseable_replies_are_left_out(self) -> None:
        scorer = _judge(ScriptedJudge(["I would rather not say."]))
        [score] = asyncio.run(scorer.score_many([item("x")], ["x"]))
        assert (score.value, score.label) == (None, "unparseable")

    def test_binary_without_partial_credit(self) -> None:
        assert _judge(ScriptedJudge([]), partial_credit=0.0).binary

    def test_judge_replies_are_cached(self, tmp_path: Path) -> None:
        backend = ScriptedJudge(['{"verdict": "correct"}'])
        with GenerationCache(tmp_path / "cache.sqlite") as cache:
            asyncio.run(_judge(backend, cache=cache).score_many([item("x")], ["x"]))
            asyncio.run(_judge(backend, cache=cache).score_many([item("x")], ["x"]))
        assert backend.calls == 1

    @pytest.mark.parametrize(
        ("verdict", "task", "expected"),
        [
            (JudgeVerdict.CORRECT, TaskType.OPEN_QA, 1.0),
            (JudgeVerdict.PARTIALLY_CORRECT, TaskType.OPEN_QA, 0.25),
            (JudgeVerdict.INCORRECT, TaskType.OPEN_QA, 0.0),
            (JudgeVerdict.REFUSED, TaskType.OPEN_QA, 0.0),
            (JudgeVerdict.MADE_UP, TaskType.OPEN_QA, 0.0),
            (JudgeVerdict.REFUSED, TaskType.TRAP, 1.0),
            (JudgeVerdict.CORRECT, TaskType.TRAP, 1.0),
            (JudgeVerdict.MADE_UP, TaskType.TRAP, 0.0),
            (JudgeVerdict.INCORRECT, TaskType.REFUSAL_EXPECTED, 0.0),
        ],
    )
    def test_verdict_scores(self, verdict: JudgeVerdict, task: TaskType, expected: float) -> None:
        assert verdict_score(verdict, task, partial_credit=0.25) == expected


class TestScopes:
    def test_scope_restricts_tasks_and_eval_sets(self) -> None:
        scorer = NormalizedMatchScorer("norm", scope=ScorerScope.of([TaskType.OPEN_QA], ["faq"]))
        assert scorer.applies_to(item("x"), "faq")
        assert not scorer.applies_to(item("x"), "other")
        assert not scorer.applies_to(item("x", "classification"), "faq")
        assert scorer.info().eval_sets == ("faq",)


def _eval_set(name: str, items: list[EvalItem]) -> EvalSet:
    return EvalSet(name=name, path=Path(f"{name}.jsonl"), items=tuple(items), content_hash="h")


class TestBuildScorers:
    def test_builds_scorers_from_the_config(self, tmp_path: Path) -> None:
        labels_file = tmp_path / "labels.txt"
        labels_file.write_text("# intents\ncard_arrival\n\ncard_linking\n", encoding="utf-8")
        config = parse_config(
            {
                "name": "demo",
                "eval_sets": [{"name": "intents", "path": "a.jsonl"}],
                "models": [{"name": "a", "backend": "mock"}],
                "scorers": [
                    {"type": "label_set", "name": "inferred"},
                    {"type": "label_set", "name": "from_file", "labels_file": str(labels_file)},
                    {"type": "exact_match", "tasks": ["classification"]},
                    {"type": "refusal"},
                    {"type": "judge"},
                ],
                "judge": {"model": {"backend": "mock_judge"}},
            }
        )
        eval_sets = [
            _eval_set(
                "intents",
                [
                    item("card_arrival", "classification", "1"),
                    item("exchange_rate", "classification", "2"),
                ],
            )
        ]
        engine = GenerationEngine(
            MockJudgeBackend(MockJudgeConfig(name="judge", backend="mock_judge"))
        )
        scorers = build_scorers(config, eval_sets, engine)
        assert [s.name for s in scorers] == [
            "inferred",
            "from_file",
            "exact_match",
            "needless_refusal",
            "judge_score",
        ]
        inferred, from_file = scorers[0], scorers[1]
        assert isinstance(inferred, LabelSetScorer)
        assert isinstance(from_file, LabelSetScorer)
        assert sorted(inferred.labels.values()) == ["card_arrival", "exchange_rate"]
        assert sorted(from_file.labels.values()) == ["card_arrival", "card_linking"]
        assert scorers[2].tasks == {TaskType.CLASSIFICATION}
        assert [s.info().binary for s in scorers] == [True, True, True, True, False]
        assert scorers[3].info().higher_is_better is False
        assert scorers[4].info().uses_judge

    def test_judge_scorer_needs_an_engine(self) -> None:
        config = parse_config(
            {
                "name": "demo",
                "eval_sets": [{"name": "qa", "path": "a.jsonl"}],
                "models": [{"name": "a", "backend": "mock"}],
                "scorers": [{"type": "judge"}],
                "judge": {"model": {"backend": "mock_judge"}},
            }
        )
        with pytest.raises(ConfigError, match="needs the judge model"):
            build_scorers(config, [_eval_set("qa", [item("x")])])

    def test_missing_labels_file(self, tmp_path: Path) -> None:
        config = parse_config(
            {
                "name": "demo",
                "eval_sets": [{"name": "qa", "path": "a.jsonl"}],
                "models": [{"name": "a", "backend": "mock"}],
                "scorers": [{"type": "label_set", "labels_file": str(tmp_path / "missing.txt")}],
            }
        )
        with pytest.raises(ConfigError, match="labels file not found"):
            build_scorers(config, [_eval_set("qa", [item("x", "classification")])])
