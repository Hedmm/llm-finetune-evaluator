"""LLM-as-judge scoring with a rubric."""

from __future__ import annotations

from collections.abc import Sequence

from llm_finetune_evaluator.backends.base import GenerationRequest
from llm_finetune_evaluator.backends.engine import GenerationEngine, ProgressCallback
from llm_finetune_evaluator.config import SamplingParams
from llm_finetune_evaluator.dataset import EvalItem, TaskType
from llm_finetune_evaluator.rubric import JudgeVerdict, build_judge_messages, parse_judge_reply
from llm_finetune_evaluator.scorers.base import Score, Scorer, ScorerScope

UNPARSEABLE = "unparseable"
"""Label of items whose judge reply had no recognisable verdict."""


def verdict_score(verdict: JudgeVerdict, task: TaskType, partial_credit: float) -> float:
    """Map a verdict to a score in [0, 1].

    For answerable items ``correct`` scores 1, ``partially_correct`` scores
    ``partial_credit`` and everything else 0; a refusal earns nothing. For trap
    and refusal_expected items abstaining is the right answer, so ``refused``
    (and ``correct``, should the judge use it for an appropriate refusal)
    scores 1.
    """
    if verdict is JudgeVerdict.PARTIALLY_CORRECT:
        return partial_credit
    if task.expects_abstention:
        return 1.0 if verdict in (JudgeVerdict.REFUSED, JudgeVerdict.CORRECT) else 0.0
    return 1.0 if verdict is JudgeVerdict.CORRECT else 0.0


class JudgeScorer(Scorer):
    """Judge score: rubric grade from a separate judge model.

    The judge sees the question, the reference and one response at a time,
    never the name of the model that wrote it. Replies without a recognisable
    verdict are labelled ``unparseable`` and left out of the statistics.

    Args:
        name: Metric name.
        engine: Generation engine of the judge model (with its cache and retries).
        sampling: Judge decoding parameters (greedy by default).
        partial_credit: Score for ``partially_correct``.
        instructions: Extra grading instructions appended to the rubric.
        scope: Optional task and eval set restriction.
    """

    type_name = "judge"
    default_tasks = frozenset({TaskType.OPEN_QA, TaskType.TRAP, TaskType.REFUSAL_EXPECTED})
    uses_judge = True

    def __init__(
        self,
        name: str,
        *,
        engine: GenerationEngine,
        sampling: SamplingParams,
        partial_credit: float = 0.5,
        instructions: str | None = None,
        scope: ScorerScope | None = None,
    ) -> None:
        super().__init__(name, scope)
        self.engine = engine
        self.sampling = sampling
        self.partial_credit = partial_credit
        self.instructions = instructions
        self.progress: ProgressCallback | None = None

    @property
    def binary(self) -> bool:
        """Binary unless partially correct answers earn partial credit."""
        return self.partial_credit in (0.0, 1.0)

    @property
    def description(self) -> str:
        """Describe the judge and the partial credit."""
        return (
            f"Judge score from {self.engine.backend.name} "
            f"(correct = 1, partially correct = {self.partial_credit:g}, otherwise 0)"
        )

    def requests(
        self, items: Sequence[EvalItem], outputs: Sequence[str]
    ) -> list[GenerationRequest]:
        """Build one judge request per (item, output) pair."""
        return [
            GenerationRequest(
                messages=build_judge_messages(item, output, self.instructions),
                sampling=self.sampling,
                item_id=item.id,
            )
            for item, output in zip(items, outputs, strict=True)
        ]

    async def verdicts(
        self, items: Sequence[EvalItem], outputs: Sequence[str]
    ) -> list[tuple[JudgeVerdict | None, str, str]]:
        """Return ``(verdict, reason, raw reply)`` for each output."""
        generations = await self.engine.generate(self.requests(items, outputs), self.progress)
        results = []
        for generation in generations:
            verdict, reason = parse_judge_reply(generation.text)
            results.append((verdict, reason, generation.text))
        return results

    async def score_many(self, items: Sequence[EvalItem], outputs: Sequence[str]) -> list[Score]:
        """Ask the judge to grade every output."""
        scores = []
        for item, (verdict, reason, raw) in zip(
            items, await self.verdicts(items, outputs), strict=True
        ):
            if verdict is None:
                scores.append(Score(None, label=UNPARSEABLE, details={"raw": raw[:500]}))
                continue
            value = verdict_score(verdict, item.task, self.partial_credit)
            scores.append(Score(value, label=verdict.value, details={"reason": reason}))
        return scores
