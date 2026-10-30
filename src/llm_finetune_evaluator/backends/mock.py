"""Deterministic simulated models for tests and offline demos.

:class:`MockBackend` simulates a model with a known skill profile. It looks up
each request's eval item (by ``item_id``) and decides, from a hash of the item
and the model seed, whether to answer correctly, partially, wrongly, refuse, or
make something up. The same inputs always give the same outputs.

Real models agree on which items are easy, so their errors are correlated; that
is why paired tests are so much more sensitive than unpaired ones. The mock
reproduces this with a Gaussian copula: each item has a latent difficulty shared
by every mock model, mixed with model-specific noise according to
``item_correlation``. Each model's overall rates still match its profile.

:class:`MockJudgeBackend` is a heuristic judge for the same purpose. It grades
by normalised string matching and refusal patterns and then flips a share of its
verdicts, so that judge calibration has disagreements to find.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from statistics import NormalDist
from typing import Any

from llm_finetune_evaluator.backends.base import Backend, Generation, GenerationRequest
from llm_finetune_evaluator.config import MockJudgeConfig, MockModelConfig, MockSkill
from llm_finetune_evaluator.dataset import EvalItem, TaskType
from llm_finetune_evaluator.errors import BackendError
from llm_finetune_evaluator.rubric import JudgeVerdict, extract_section
from llm_finetune_evaluator.text import (
    RefusalDetector,
    contains_phrase,
    normalize_text,
    token_f1,
)

_NORMAL = NormalDist()

_REFUSALS = (
    "I'm sorry, but I can't help with that question.",
    "I don't have enough information to answer that.",
    "I cannot answer that question.",
)
_TRAP_ABSTENTIONS = (
    "The source does not mention this.",
    "That information is not provided in the source.",
    "I don't know: the text doesn't say.",
)
_DECLINES = (
    "I can't help with that request.",
    "I'm sorry, but I cannot share that.",
)


class MockOutcome(str, Enum):
    """What kind of answer the mock produced; the ground truth for demos."""

    CORRECT = "correct"
    PARTIAL = "partial"
    WRONG = "wrong"
    REFUSED = "refused"
    ABSTAINED = "abstained"
    MADE_UP = "made_up"
    COMPLIED = "complied"
    SCRIPTED = "scripted"


@dataclass(frozen=True)
class SkillProfile:
    """Probabilities that drive the simulated answers.

    Attributes:
        accuracy: Chance of a correct answer on an answerable item.
        partial_rate: Chance of a partially correct answer, where one exists.
        refusal_rate: Chance of refusing an answerable item.
        abstain_rate: Chance of abstaining on a trap or refusal_expected item.
    """

    accuracy: float
    partial_rate: float
    refusal_rate: float
    abstain_rate: float

    def updated(self, skill: MockSkill) -> SkillProfile:
        """Return a copy with every rate that ``skill`` sets replaced."""
        changes = {key: value for key, value in skill.model_dump().items() if value is not None}
        return replace(self, **changes)


@dataclass(frozen=True)
class SimulatedAnswer:
    """A simulated model output and the outcome it represents."""

    text: str
    outcome: MockOutcome


def _uniform(*parts: object) -> float:
    """Return a deterministic number in [0, 1) derived from ``parts``."""
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _standard_normal(*parts: object) -> float:
    return _NORMAL.inv_cdf(min(max(_uniform(*parts), 1e-12), 1.0 - 1e-12))


def _choose(options: Sequence[str], u: float) -> str:
    return options[min(len(options) - 1, int(u * len(options)))]


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _answer_pool(answers: Iterable[str]) -> list[tuple[str, str]]:
    """Return distinct answers with their normalised form, in a stable order."""
    return [(normalize_text(answer), answer) for answer in sorted(set(answers))]


def default_mock_seed(name: str) -> int:
    """Seed used for a mock model without an explicit seed: a hash of its name."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")


class MockSimulator:
    """Decides what a simulated model answers for each item.

    Args:
        config: The mock model's settings.
        items: Every eval item; used to look items up and to draw plausible
            wrong answers from other items' references.
    """

    def __init__(self, config: MockModelConfig, items: Iterable[EvalItem]) -> None:
        self.config = config
        self.seed = config.seed if config.seed is not None else default_mock_seed(config.name)
        self._base = SkillProfile(
            accuracy=config.accuracy,
            partial_rate=config.partial_rate,
            refusal_rate=config.refusal_rate,
            abstain_rate=config.abstain_rate,
        )
        items = list(items)
        self._labels = _answer_pool(
            str(answer)
            for item in items
            if item.task is TaskType.CLASSIFICATION
            for answer in item.references()
        )
        self._answers = _answer_pool(
            answer
            for item in items
            if item.task is TaskType.OPEN_QA
            for answer in item.references()
            if isinstance(answer, str)
        )

    def profile_for(self, item: EvalItem) -> SkillProfile:
        """Return the skill profile for ``item`` after applying ``by_tag`` overrides."""
        profile = self._base
        for key, skill in self.config.by_tag.items():
            dimension, _, value = key.partition("=")
            actual = item.task.value if dimension == "task" else item.tags.get(dimension)
            if actual == value:
                profile = profile.updated(skill)
        return profile

    def answer(self, item: EvalItem) -> SimulatedAnswer:
        """Return the simulated answer to ``item``."""
        scripted = self.config.responses.get(item.id)
        if scripted is not None:
            return SimulatedAnswer(scripted, MockOutcome.SCRIPTED)
        profile = self.profile_for(item)
        rho = self.config.item_correlation
        latent = math.sqrt(rho) * _standard_normal("difficulty", item.id) + math.sqrt(
            1.0 - rho
        ) * _standard_normal("model", self.seed, item.id)
        u = _NORMAL.cdf(latent)
        pick = _uniform("pick", self.seed, item.id)
        if item.task.expects_abstention:
            if u < profile.abstain_rate:
                options = _TRAP_ABSTENTIONS if item.task is TaskType.TRAP else _DECLINES
                return SimulatedAnswer(_choose(options, pick), MockOutcome.ABSTAINED)
            invented = self._pick_other(self._answers, set(), pick) or "It is 42."
            if item.task is TaskType.TRAP:
                return SimulatedAnswer(invented, MockOutcome.MADE_UP)
            return SimulatedAnswer(f"Sure. Here it is: {invented}", MockOutcome.COMPLIED)
        if _uniform("refusal", self.seed, item.id) < profile.refusal_rate:
            return SimulatedAnswer(_choose(_REFUSALS, pick), MockOutcome.REFUSED)
        if u < profile.accuracy:
            return SimulatedAnswer(self._correct(item), MockOutcome.CORRECT)
        if u < profile.accuracy + profile.partial_rate:
            partial = self._partial(item)
            if partial is not None:
                return SimulatedAnswer(partial, MockOutcome.PARTIAL)
        return SimulatedAnswer(self._wrong(item, pick), MockOutcome.WRONG)

    @staticmethod
    def _pick_other(options: Sequence[tuple[str, str]], exclude: set[str], u: float) -> str | None:
        """Pick an answer whose normalised form is not in ``exclude``, starting at ``u``."""
        start = int(u * len(options))
        for offset in range(len(options)):
            normalized, original = options[(start + offset) % len(options)]
            if normalized not in exclude:
                return original
        return None

    @staticmethod
    def _correct(item: EvalItem) -> str:
        reference = item.references()[0] if item.task is not TaskType.EXTRACTION else item.reference
        if item.task is TaskType.EXTRACTION:
            return json.dumps(reference, ensure_ascii=False)
        if isinstance(reference, (int, float)):
            return _format_number(reference)
        return str(reference)

    @staticmethod
    def _partial(item: EvalItem) -> str | None:
        if item.task is TaskType.EXTRACTION and len(item.reference) > 1:
            kept = dict(list(item.reference.items())[:-1])
            return json.dumps(kept, ensure_ascii=False)
        reference = item.references()[0]
        if item.task is TaskType.OPEN_QA and isinstance(reference, str):
            words = reference.split()
            if len(words) >= 2:
                return " ".join(words[: math.ceil(len(words) / 2)])
        return None

    def _wrong(self, item: EvalItem, pick: float) -> str:
        if item.task is TaskType.EXTRACTION:
            return json.dumps(_corrupt_fields(item.reference), ensure_ascii=False)
        references = item.references()
        excluded = {normalize_text(str(answer)) for answer in references}
        if item.task is TaskType.CLASSIFICATION:
            return self._pick_other(self._labels, excluded, pick) or "unknown"
        first = references[0]
        if isinstance(first, (int, float)):
            return _format_number(round(first * 1.5 + 1, 2))
        return self._pick_other(self._answers, excluded, pick) or "I am not sure, maybe 42."


def _corrupt_fields(reference: dict[str, Any]) -> dict[str, Any]:
    """Keep the first field and get every other field wrong."""
    corrupted: dict[str, Any] = {}
    for position, (key, value) in enumerate(reference.items()):
        if position == 0:
            corrupted[key] = value
        elif isinstance(value, bool):
            corrupted[key] = not value
        elif isinstance(value, (int, float)):
            corrupted[key] = value + 1
        elif isinstance(value, str):
            corrupted[key] = "unknown"
        else:
            corrupted[key] = None
    return corrupted


class MockBackend(Backend):
    """Serves :class:`MockSimulator` answers through the backend interface.

    Args:
        config: The mock model's settings.
        items: Eval items by id; every request must carry a known ``item_id``.
    """

    kind = "mock"

    def __init__(self, config: MockModelConfig, items: Mapping[str, EvalItem]) -> None:
        super().__init__(config.name, max_concurrency=config.max_concurrency)
        self.config = config
        self.simulator = MockSimulator(config, items.values())
        self._items = items
        self.calls = 0

    def identity(self) -> dict[str, Any]:
        """Return the simulated skill profile and seed."""
        settings = self.config.model_dump(
            mode="json", exclude={"name", "sampling", "max_concurrency", "latency_s"}
        )
        return {"backend": self.kind, **settings, "seed": self.simulator.seed}

    def request_identity(self, request: GenerationRequest) -> dict[str, Any]:
        """Mock answers depend on the item, not only on the prompt."""
        return {"item_id": request.item_id}

    async def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        """Answer each request from the simulator."""
        generations = []
        for request in requests:
            item = self._items.get(request.item_id or "")
            if item is None:
                raise BackendError(f"{self.name}: mock backend has no item {request.item_id!r}")
            if self.config.latency_s:
                await asyncio.sleep(self.config.latency_s)
            self.calls += 1
            generations.append(
                Generation(
                    text=self.simulator.answer(item).text,
                    finish_reason="stop",
                    latency_s=self.config.latency_s,
                )
            )
        return generations


_SIMULATED_MISTAKES = {
    JudgeVerdict.CORRECT: JudgeVerdict.PARTIALLY_CORRECT,
    JudgeVerdict.PARTIALLY_CORRECT: JudgeVerdict.CORRECT,
    JudgeVerdict.INCORRECT: JudgeVerdict.PARTIALLY_CORRECT,
    JudgeVerdict.REFUSED: JudgeVerdict.INCORRECT,
    JudgeVerdict.MADE_UP: JudgeVerdict.INCORRECT,
}


class MockJudgeBackend(Backend):
    """A heuristic judge that answers judge prompts with JSON verdicts.

    Args:
        config: The mock judge's settings.
    """

    kind = "mock_judge"

    def __init__(self, config: MockJudgeConfig) -> None:
        super().__init__(config.name, max_concurrency=config.max_concurrency)
        self.noise = config.noise
        self.seed = config.seed
        self._detector = RefusalDetector()
        self.calls = 0

    def identity(self) -> dict[str, Any]:
        """Return the noise level and seed."""
        return {"backend": self.kind, "noise": self.noise, "seed": self.seed}

    def grade(self, prompt: str) -> tuple[JudgeVerdict, str]:
        """Grade a prompt built by :func:`~llm_finetune_evaluator.rubric.build_judge_messages`."""
        task_name = extract_section(prompt, "task")
        response = extract_section(prompt, "response")
        reference = extract_section(prompt, "reference")
        if task_name is None or response is None or reference is None:
            raise BackendError(
                f"{self.name}: mock judge received a prompt that is not a judge prompt"
            )
        task = TaskType(task_name)
        if self._detector.is_refusal(response):
            verdict, reason = JudgeVerdict.REFUSED, "The response declines to answer."
        elif task.expects_abstention:
            verdict, reason = (
                JudgeVerdict.MADE_UP,
                "The response answers a question it cannot answer.",
            )
        else:
            answers = [line[2:] for line in reference.splitlines() if line.startswith("- ")]
            if any(contains_phrase(response, answer) for answer in answers):
                verdict, reason = (
                    JudgeVerdict.CORRECT,
                    "The response contains the reference answer.",
                )
            elif max((token_f1(response, answer) for answer in answers), default=0.0) >= 0.5:
                verdict, reason = (
                    JudgeVerdict.PARTIALLY_CORRECT,
                    "The response overlaps the reference.",
                )
            else:
                verdict, reason = (
                    JudgeVerdict.INCORRECT,
                    "The response does not match the reference.",
                )
        if _uniform("judge-noise", self.seed, prompt) < self.noise:
            verdict = _SIMULATED_MISTAKES[verdict]
            reason = "Simulated judge mistake."
        return verdict, reason

    async def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        """Grade each judge prompt."""
        generations = []
        for request in requests:
            self.calls += 1
            verdict, reason = self.grade(request.messages[-1].content)
            text = json.dumps({"verdict": verdict.value, "reason": reason})
            generations.append(Generation(text=text, finish_reason="stop"))
        return generations
