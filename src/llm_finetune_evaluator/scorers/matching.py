"""String matching scorers: exact, normalised, label set and regular expression."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import lru_cache

from llm_finetune_evaluator.dataset import ANSWERABLE_TASKS, EvalItem, TaskType
from llm_finetune_evaluator.errors import ScorerError
from llm_finetune_evaluator.scorers.base import Score, ScorerScope, SyncScorer
from llm_finetune_evaluator.text import contains_phrase, normalize_text

_TEXT_TASKS = frozenset({TaskType.CLASSIFICATION, TaskType.OPEN_QA})


class ExactMatchScorer(SyncScorer):
    """Exact match: the output equals a reference after trimming whitespace.

    Args:
        name: Metric name.
        case_sensitive: Compare case-sensitively (the default).
        scope: Optional task and eval set restriction.
    """

    type_name = "exact_match"
    default_tasks = _TEXT_TASKS

    def __init__(
        self, name: str, *, case_sensitive: bool = True, scope: ScorerScope | None = None
    ) -> None:
        super().__init__(name, scope)
        self.case_sensitive = case_sensitive

    def score(self, item: EvalItem, output: str) -> Score:
        """Return 1 if the trimmed output equals any reference."""
        candidate = output.strip()
        references = [str(reference).strip() for reference in item.references()]
        if not self.case_sensitive:
            candidate = candidate.casefold()
            references = [reference.casefold() for reference in references]
        return Score(float(candidate in references))


class NormalizedMatchScorer(SyncScorer):
    """Normalised match: equal after lowercasing and removing punctuation and articles.

    Args:
        name: Metric name.
        mode: ``"exact"`` requires the normalised strings to be equal;
            ``"contains"`` accepts an output that contains a reference as a
            whole-word phrase ("It takes five days." contains "five days").
        scope: Optional task and eval set restriction.
    """

    type_name = "normalized_match"
    default_tasks = _TEXT_TASKS

    def __init__(self, name: str, *, mode: str = "exact", scope: ScorerScope | None = None) -> None:
        super().__init__(name, scope)
        if mode not in ("exact", "contains"):
            raise ValueError(f"mode must be 'exact' or 'contains', got {mode!r}")
        self.mode = mode

    @property
    def description(self) -> str:
        """Describe the matching mode."""
        if self.mode == "contains":
            return "Normalised match: the output contains a reference answer"
        return "Normalised match: equal to a reference after normalisation"

    def score(self, item: EvalItem, output: str) -> Score:
        """Return 1 if the output matches any reference in the configured mode."""
        references = [str(reference) for reference in item.references()]
        if self.mode == "contains":
            matched = any(contains_phrase(output, reference) for reference in references)
        else:
            normalized = normalize_text(output)
            matched = any(normalized == normalize_text(reference) for reference in references)
        return Score(float(matched))


_LABEL_PREFIX = re.compile(
    r"^(?:the\s+)?(?:label|intent|category|class|answer)(?:\s+is)?\s*[:=\-]?\s*", re.IGNORECASE
)
_WRAPPING = " \t\"'`*._:"
_LIST_SEPARATORS = re.compile(r"[,;\n]")


def label_key(text: str) -> str:
    """Return the comparison key of a label ("Card_Arrival" and "card arrival" agree)."""
    return normalize_text(text)


class LabelSetScorer(SyncScorer):
    """Label accuracy over a closed label set; answers outside the set count as wrong.

    The first non-empty line of the output is taken as the prediction, after
    removing quotes, Markdown emphasis and prefixes such as "Intent:". It is
    matched against the label set ignoring case and the difference between
    spaces, hyphens and underscores. Predictions that are not in the set are
    labelled ``invalid_label``, which reports show as the valid-label rate.

    With ``multi_label`` the output is split on commas, semicolons and line
    breaks, and the predicted set must equal the reference set.

    Args:
        name: Metric name.
        labels: The allowed labels.
        multi_label: Score label sets instead of single labels.
        scope: Optional task and eval set restriction.

    Raises:
        ScorerError: If two labels have the same comparison key.
    """

    type_name = "label_set"
    default_tasks = frozenset({TaskType.CLASSIFICATION})

    def __init__(
        self,
        name: str,
        *,
        labels: Iterable[str],
        multi_label: bool = False,
        scope: ScorerScope | None = None,
    ) -> None:
        super().__init__(name, scope)
        self.multi_label = multi_label
        self.labels: dict[str, str] = {}
        for label in labels:
            key = label_key(label)
            if key in self.labels and self.labels[key] != label:
                raise ScorerError(
                    f"labels {self.labels[key]!r} and {label!r} "
                    "are indistinguishable after normalisation"
                )
            self.labels[key] = label
        if not self.labels:
            raise ScorerError(f"scorer {name!r}: the label set is empty")

    @property
    def description(self) -> str:
        """Describe the label set size."""
        kind = "Label-set accuracy" if not self.multi_label else "Exact label-set match"
        return f"{kind} over {len(self.labels)} labels (off-list answers count as wrong)"

    def parse(self, output: str) -> str | None:
        """Map an output to a label of the set, or ``None`` if it is not one."""
        lines = [line for line in output.splitlines() if line.strip()]
        if not lines:
            return None
        candidate = _LABEL_PREFIX.sub("", lines[0].strip()).strip(_WRAPPING)
        return self.labels.get(label_key(candidate))

    def _reference_labels(self, item: EvalItem) -> list[str]:
        references = [str(reference) for reference in item.references()]
        unknown = [reference for reference in references if label_key(reference) not in self.labels]
        if unknown:
            raise ScorerError(
                f"item {item.id!r}: reference labels {unknown} "
                f"are not in the label set of {self.name!r}"
            )
        return [self.labels[label_key(reference)] for reference in references]

    def score(self, item: EvalItem, output: str) -> Score:
        """Score a single label or a label set."""
        references = self._reference_labels(item)
        if not self.multi_label:
            predicted = self.parse(output)
            if predicted is None:
                return Score(0.0, label="invalid_label", details={"predicted": None})
            return Score(float(predicted in references), details={"predicted": predicted})
        parts = [
            part.strip(_WRAPPING)
            for part in _LIST_SEPARATORS.split(output)
            if part.strip(_WRAPPING)
        ]
        mapped = [self.labels.get(label_key(part)) for part in parts]
        predicted_set = {label for label in mapped if label is not None}
        expected = set(references)
        union = predicted_set | expected
        details = {
            "predicted": sorted(predicted_set),
            "jaccard": len(predicted_set & expected) / len(union) if union else 1.0,
        }
        label = "invalid_label" if None in mapped else None
        return Score(float(predicted_set == expected), label=label, details=details)


@lru_cache(maxsize=4096)
def _compile(pattern: str, flags: int) -> re.Pattern[str]:
    return re.compile(pattern, flags)


class RegexScorer(SyncScorer):
    """Regex match: the output matches a regular expression.

    Args:
        name: Metric name.
        pattern: A fixed pattern for every item, or ``None`` to use each
            item's reference answers as patterns.
        ignore_case: Match case-insensitively.
        full_match: Require the whole trimmed output to match.
        scope: Optional task and eval set restriction.
    """

    type_name = "regex"
    default_tasks = ANSWERABLE_TASKS

    def __init__(
        self,
        name: str,
        *,
        pattern: str | None = None,
        ignore_case: bool = False,
        full_match: bool = False,
        scope: ScorerScope | None = None,
    ) -> None:
        super().__init__(name, scope)
        self.pattern = pattern
        self.flags = re.IGNORECASE if ignore_case else 0
        self.full_match = full_match

    @property
    def description(self) -> str:
        """Describe where the pattern comes from."""
        source = f"/{self.pattern}/" if self.pattern is not None else "the reference pattern"
        return f"Regex match: the output {'fully ' if self.full_match else ''}matches {source}"

    def _patterns(self, item: EvalItem) -> Sequence[re.Pattern[str]]:
        sources = (
            [self.pattern] if self.pattern is not None else [str(r) for r in item.references()]
        )
        try:
            return [_compile(source, self.flags) for source in sources]
        except re.error as error:
            raise ScorerError(f"item {item.id!r}: invalid regular expression ({error})") from error

    def score(self, item: EvalItem, output: str) -> Score:
        """Return 1 if any pattern matches."""
        text = output.strip()
        matched = any(
            (pattern.fullmatch(text) if self.full_match else pattern.search(text)) is not None
            for pattern in self._patterns(item)
        )
        return Score(float(matched))
