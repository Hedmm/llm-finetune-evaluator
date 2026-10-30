"""Numeric answers within a tolerance."""

from __future__ import annotations

import math
from typing import Any

from llm_finetune_evaluator.dataset import EvalItem, TaskType
from llm_finetune_evaluator.scorers.base import Score, ScorerScope, SyncScorer
from llm_finetune_evaluator.text import extract_numbers


def numeric_references(item: EvalItem) -> list[float]:
    """Return the item's references that are numbers or numeric strings."""
    numbers = []
    for reference in item.references():
        value: Any = reference
        if isinstance(value, bool):
            continue
        if isinstance(value, str):
            parsed = extract_numbers(value)
            if len(parsed) != 1 or not value.strip():
                continue
            value = parsed[0]
        if isinstance(value, (int, float)):
            numbers.append(float(value))
    return numbers


class NumericScorer(SyncScorer):
    """Numeric match: the number in the output is within tolerance of the reference.

    Applies only to items whose reference is a number (or a string holding
    exactly one number), so it can sit next to text scorers on a mixed set.

    Args:
        name: Metric name.
        abs_tol: Absolute tolerance.
        rel_tol: Relative tolerance.
        extract: Which number of the output to use: ``"first"``, ``"last"``
            (the default, since models tend to state the result at the end) or
            ``"only"`` (the output must contain exactly one number).
        scope: Optional task and eval set restriction.
    """

    type_name = "numeric"
    default_tasks = frozenset({TaskType.OPEN_QA})

    def __init__(
        self,
        name: str,
        *,
        abs_tol: float = 0.0,
        rel_tol: float = 1e-9,
        extract: str = "last",
        scope: ScorerScope | None = None,
    ) -> None:
        super().__init__(name, scope)
        if extract not in ("first", "last", "only"):
            raise ValueError(f"extract must be 'first', 'last' or 'only', got {extract!r}")
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.extract = extract

    @property
    def description(self) -> str:
        """Describe the tolerance."""
        parts = []
        if self.abs_tol:
            parts.append(f"±{self.abs_tol:g}")
        if self.rel_tol > 1e-9:
            parts.append(f"±{self.rel_tol:.0%}")
        tolerance = " or ".join(parts) if parts else "exact"
        return f"Numeric match ({tolerance}, {self.extract} number in the output)"

    def applies_to(self, item: EvalItem, eval_set: str) -> bool:
        """Only items with a numeric reference."""
        return super().applies_to(item, eval_set) and bool(numeric_references(item))

    def score(self, item: EvalItem, output: str) -> Score:
        """Extract a number from the output and compare it with the references."""
        numbers = extract_numbers(output)
        if not numbers:
            return Score(0.0, label="no_number")
        if self.extract == "only" and len(numbers) != 1:
            return Score(0.0, label="ambiguous", details={"numbers": numbers})
        predicted = numbers[0] if self.extract == "first" else numbers[-1]
        matched = any(
            math.isclose(predicted, reference, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            for reference in numeric_references(item)
        )
        return Score(float(matched), details={"predicted": predicted})
