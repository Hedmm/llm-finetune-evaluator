"""Field-level precision, recall and F1 for JSON extraction."""

from __future__ import annotations

import math
import unicodedata
from typing import Any

from llm_finetune_evaluator.dataset import EvalItem, TaskType, canonical_json
from llm_finetune_evaluator.scorers.base import Score, ScorerScope, SyncScorer
from llm_finetune_evaluator.text import extract_json_object

_MISSING = object()


def flatten_fields(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested objects into dotted field paths.

    ``{"customer": {"name": "Ana"}, "items": [1, 2]}`` becomes
    ``{"customer.name": "Ana", "items": [1, 2]}``. Lists are kept as leaves and
    compared as unordered collections.
    """
    flat: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}{key}"
        if isinstance(child, dict) and child:
            flat.update(flatten_fields(child, f"{path}."))
        else:
            flat[path] = child
    return flat


def _is_empty(value: Any) -> bool:
    return value is _MISSING or value is None or value == "" or value == [] or value == {}


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return None
    return None


def _normalise_string(value: str, case_sensitive: bool) -> str:
    text = " ".join(unicodedata.normalize("NFKC", value).split()).strip(" .")
    return text if case_sensitive else text.casefold()


class JsonFieldsScorer(SyncScorer):
    """Field F1: field-level F1 of the JSON object in the output against the reference.

    Fields are compared after flattening nested objects into dotted paths.
    A field counts as a true positive when both sides have it with a matching
    value, as a false negative when the reference has it but the output does
    not (or has a different value), and as a false positive when the output has
    it but the reference does not (or has a different value). Null and empty
    values count as absent. Strings match after whitespace and case
    normalisation, numbers (also numeric strings) within ``numeric_tolerance``,
    and lists as unordered collections.

    The item score is the F1; precision, recall and the counts are kept in the
    score details, and reports add micro-averaged precision, recall and F1.
    An output without a JSON object is labelled ``invalid_json``.

    Args:
        name: Metric name.
        numeric_tolerance: Absolute tolerance for numeric fields.
        case_sensitive: Compare strings case-sensitively.
        scope: Optional task and eval set restriction.
    """

    type_name = "json_fields"
    default_tasks = frozenset({TaskType.EXTRACTION})

    def __init__(
        self,
        name: str,
        *,
        numeric_tolerance: float = 1e-6,
        case_sensitive: bool = False,
        scope: ScorerScope | None = None,
    ) -> None:
        super().__init__(name, scope)
        self.numeric_tolerance = numeric_tolerance
        self.case_sensitive = case_sensitive

    @property
    def binary(self) -> bool:
        """F1 takes values between 0 and 1."""
        return False

    def values_match(self, expected: Any, actual: Any) -> bool:
        """Whether two field values agree."""
        expected_number, actual_number = _as_number(expected), _as_number(actual)
        if expected_number is not None and actual_number is not None:
            return math.isclose(
                expected_number, actual_number, rel_tol=1e-9, abs_tol=self.numeric_tolerance
            )
        if isinstance(expected, str) and isinstance(actual, str):
            return _normalise_string(expected, self.case_sensitive) == _normalise_string(
                actual, self.case_sensitive
            )
        if isinstance(expected, list) and isinstance(actual, list):
            return sorted(map(self._canonical, expected)) == sorted(map(self._canonical, actual))
        return bool(expected == actual)

    def _canonical(self, value: Any) -> str:
        number = _as_number(value)
        if number is not None:
            return repr(number)
        if isinstance(value, str):
            return _normalise_string(value, self.case_sensitive)
        return canonical_json(value)

    def score(self, item: EvalItem, output: str) -> Score:
        """Compare the output's JSON object with the reference field by field."""
        expected = {k: v for k, v in flatten_fields(item.reference).items() if not _is_empty(v)}
        parsed = extract_json_object(output)
        if parsed is None:
            f1 = 0.0 if expected else 1.0
            invalid: dict[str, Any] = {
                "precision": f1,
                "recall": f1,
                "f1": f1,
                "tp": 0,
                "fp": 0,
                "fn": len(expected),
            }
            return Score(f1, label="invalid_json", details=invalid)
        actual = {k: v for k, v in flatten_fields(parsed).items() if not _is_empty(v)}
        tp = fp = fn = 0
        wrong: list[str] = []
        missing: list[str] = []
        extra: list[str] = []
        for field in sorted(set(expected) | set(actual)):
            if field in expected and field in actual:
                if self.values_match(expected[field], actual[field]):
                    tp += 1
                else:
                    fp += 1
                    fn += 1
                    wrong.append(field)
            elif field in expected:
                fn += 1
                missing.append(field)
            else:
                fp += 1
                extra.append(field)
        precision = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
        recall = tp / (tp + fn) if tp + fn else (1.0 if tp + fp == 0 else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        details: dict[str, Any] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        if wrong:
            details["wrong"] = wrong
        if missing:
            details["missing"] = missing
        if extra:
            details["extra"] = extra
        return Score(f1, details=details)
