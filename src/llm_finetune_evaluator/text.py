"""Text utilities shared by scorers and backends.

Normalisation, refusal detection, and extraction of JSON objects and numbers
from free-form model output.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
_DECIMAL_PLACEHOLDER = "\x00"
_ASCII_PUNCTUATION = frozenset(string.punctuation)
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2212": "-"})


_WORD_SEPARATORS = frozenset("-_/")


def _is_punctuation(char: str) -> bool:
    return char in _ASCII_PUNCTUATION or unicodedata.category(char).startswith("P")


def normalize_text(text: str) -> str:
    """Normalise text for lenient comparison.

    Follows the SQuAD answer normalisation (lowercase, drop punctuation and the
    articles "a", "an" and "the", collapse whitespace) with three refinements:
    Unicode is NFKC-normalised; "-", "_" and "/" separate words instead of
    being deleted, so the label "card_arrival" matches "card arrival"; and
    decimal points inside numbers are kept, so "3.5" does not become "35".
    Thousands separators are removed.

    Args:
        text: Any string.

    Returns:
        The normalised string.
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).casefold()
    text = _THOUSANDS_SEPARATOR.sub("", text)
    text = _DECIMAL_POINT.sub(_DECIMAL_PLACEHOLDER, text)
    kept = []
    for char in text:
        if char in _WORD_SEPARATORS:
            kept.append(" ")
        elif not _is_punctuation(char):
            kept.append(char)
    text = "".join(kept).replace(_DECIMAL_PLACEHOLDER, ".")
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    """Return the normalised tokens of ``text``."""
    return normalize_text(text).split()


def contains_phrase(haystack: str, needle: str) -> bool:
    """Whether the normalised ``needle`` occurs in ``haystack`` on token boundaries."""
    normalized_needle = normalize_text(needle)
    if not normalized_needle:
        return False
    return f" {normalized_needle} " in f" {normalize_text(haystack)} "


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token overlap F1 between two strings."""
    predicted = tokens(prediction)
    expected = tokens(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^\s*<think>(?!.*</think>).*", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks emitted by reasoning models.

    A reasoning block that was cut off before ``</think>`` (the model ran out of
    tokens while thinking) leaves no answer, so the result is empty.
    """
    without_blocks = _THINK_BLOCK.sub("", text)
    return _UNCLOSED_THINK.sub("", without_blocks).strip()


REFUSAL_PATTERNS: tuple[str, ...] = (
    # "I can't help with that", "We are unable to provide ...".
    r"\b(?:i|we)\s+(?:can ?not|can't|am unable to|'m unable to|are unable to|won't|will not"
    r"|am not able to|'m not able to)\s+(?:help|assist|answer|provide|share|find|determine"
    r"|say|tell|confirm|locate|comply|give|disclose|reveal|do that)\b",
    # "I don't know", "I do not have enough information".
    r"\b(?:i|we)\s+(?:do not|don't|did not|didn't)\s+(?:know|have\s+"
    r"(?:enough\s+|sufficient\s+|any\s+)?(?:information|details|data|access|knowledge|context))\b",
    # "The source does not mention ...", "the provided text doesn't say".
    r"\b(?:source|context|document|documentation|text|passage|article|material|excerpt"
    r"|information)s?\b[^.!?\n]{0,60}?\b(?:does not|doesn't|do not|don't|did not|didn't)\s+"
    r"(?:say|state|mention|specify|contain|include|provide|cover|indicate|give|describe"
    r"|address|discuss|list)\b",
    # "This is not mentioned in the source", "isn't specified in the text".
    r"\b(?:is|are|was|were|isn't|aren't|wasn't|weren't)\s+(?:not\s+)?(?:mentioned|stated"
    r"|specified|provided|included|covered|given|described|addressed|discussed|listed)\s+in\s+"
    r"the\s+(?:source|context|document|documentation|text|passage|article|material|excerpt"
    r"|information)\b",
    r"\bnot\s+(?:mentioned|stated|specified|provided|covered|given|available)\s+in\s+the\s+"
    r"(?:source|context|document|documentation|text|passage|article|material|excerpt"
    r"|provided)\b",
    # "No information about ...".
    r"\bno\s+(?:information|mention|details|data|indication|record)\s+(?:about|on|of|regarding"
    r"|is|was|in|for)\b",
    r"\b(?:cannot|can't|can not)\s+be\s+(?:answered|determined|found|inferred|confirmed)\b",
    r"\bunable\s+to\s+(?:answer|help|assist|determine|find|provide|confirm)\b",
    r"\binsufficient\s+(?:information|context|data|details)\b",
    r"\bnot\s+in\s+the\s+(?:source|context|document|text|passage|provided)\b",
    r"\bi(?:'m| am)\s+(?:sorry|afraid)\b[^.!?\n]{0,40}?\b(?:can't|cannot|can not|unable"
    r"|not able)\b",
)
"""Regular expressions (case-insensitive) that signal a refusal or an abstention.

They aim at a low false-positive rate: a phrase such as "not available" alone is
not enough, because it often occurs in genuine answers ("the feature is not
available on the basic plan"). Pattern matching is a heuristic; use the LLM
judge, calibrated on human labels, when refusal detection matters.
"""


class RefusalDetector:
    """Detects refusals and "the source does not say" answers with patterns.

    Args:
        extra_patterns: Additional regular expressions, for example for another
            language. They are matched case-insensitively.
    """

    def __init__(self, extra_patterns: Iterable[str] = ()) -> None:
        self._patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in (*REFUSAL_PATTERNS, *extra_patterns)
        ]

    def find(self, text: str) -> str | None:
        """Return the first refusal phrase in ``text``, or ``None``."""
        normalized = unicodedata.normalize("NFKC", text).translate(_QUOTES)
        for pattern in self._patterns:
            match = pattern.search(normalized)
            if match:
                return match.group(0)
        return None

    def is_refusal(self, text: str) -> bool:
        """Whether ``text`` declines to answer or says the answer is not available."""
        return self.find(text) is not None


_FENCED_BLOCK = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object in ``text``, or ``None``.

    Accepts bare JSON, JSON inside a Markdown code fence, and JSON surrounded by
    prose ("Here is the result: {...}").
    """
    stripped = text.strip()
    candidates = [match.group(1).strip() for match in _FENCED_BLOCK.finditer(stripped)]
    candidates.append(stripped)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            value, _ = decoder.raw_decode(stripped, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"|(?<![\w.])[-+]?\.\d+(?:[eE][-+]?\d+)?"
)


def extract_numbers(text: str) -> list[float]:
    """Return every number in ``text`` in order of appearance.

    Handles signs, thousands separators ("1,234.5"), decimals and scientific
    notation. Currency symbols and percent signs are ignored ("$12%" gives 12).
    """
    normalized = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    return [float(match.group(0).replace(",", "")) for match in _NUMBER.finditer(normalized)]
