from __future__ import annotations

import pytest

from llm_finetune_evaluator.text import (
    RefusalDetector,
    contains_phrase,
    extract_json_object,
    extract_numbers,
    normalize_text,
    strip_think_tags,
    token_f1,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Answer!", "answer"),
        ("  Paris.  ", "paris"),
        ("card_arrival", "card arrival"),
        ("Card-Arrival", "card arrival"),
        ("U.S.A.", "usa"),
        ("3.5%", "3.5"),
        ("1,000 dollars", "1000 dollars"),
        ("\u201cquoted\u201d \u2018text\u2019", "quoted text"),
        ("\ufb01ve", "five"),
        ("a cat and an apple", "cat and apple"),
    ],
)
def test_normalize_text(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


def test_contains_phrase_respects_token_boundaries() -> None:
    assert contains_phrase("It takes 5 days.", "5 days")
    assert not contains_phrase("It takes 15 days.", "5 days")
    assert contains_phrase("Refunds take FIVE business days", "five business days")
    assert not contains_phrase("anything", "")


def test_token_f1() -> None:
    assert token_f1("five business days", "five business days") == 1.0
    assert token_f1("five days", "five business days") == pytest.approx(0.8)
    assert token_f1("no overlap", "five business days") == 0.0
    assert token_f1("", "") == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>Let me reason.</think>Paris", "Paris"),
        ("<think>a</think> Paris <think>b</think>", "Paris"),
        ("<think>never finished because max_tokens ran out", ""),
        ("No reasoning here.", "No reasoning here."),
    ],
)
def test_strip_think_tags(text: str, expected: str) -> None:
    assert strip_think_tags(text) == expected


REFUSALS = [
    "I'm sorry, but I can't help with that question.",
    "I don't have enough information to answer that.",
    "I cannot answer that question.",
    "The source does not mention this.",
    "That information is not provided in the source.",
    "I don't know: the text doesn't say.",
    "I can't help with that request.",
    "I'm sorry, but I cannot share that.",
    "The provided context doesn't specify the warranty period.",
    "This is not mentioned in the document.",
    "There is no information about pricing in the passage.",
    "The answer cannot be determined from the text.",
    "Unfortunately, I am unable to provide that.",
    "I\u2019m afraid I can\u2019t disclose internal credentials.",
    "Insufficient information to answer.",
    "That is not in the source.",
    "We are unable to help with this.",
]

ANSWERS = [
    "Refunds take five business days.",
    "Sorry for the delay: your card ships in 3 days.",
    "The feature is not available on the basic plan.",
    "No, you cannot transfer more than 500 euros per day.",
    "I can confirm the fee is 5 dollars.",
    "You don't need a PIN for contactless payments under 50 euros.",
    "The card is not activated until you sign in once.",
    "Yes. The text says the warranty lasts two years.",
    "There is no fee for the first transfer each month.",
]


@pytest.mark.parametrize("text", REFUSALS)
def test_refusal_detector_finds_refusals(text: str) -> None:
    assert RefusalDetector().is_refusal(text)


@pytest.mark.parametrize("text", ANSWERS)
def test_refusal_detector_ignores_genuine_answers(text: str) -> None:
    assert RefusalDetector().find(text) is None


def test_refusal_detector_accepts_extra_patterns() -> None:
    detector = RefusalDetector(extra_patterns=[r"\bich wei(?:ss|\u00df) es nicht\b"])
    assert detector.is_refusal("Ich wei\u00df es nicht.")
    assert not RefusalDetector().is_refusal("Ich wei\u00df es nicht.")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": {"b": [1, 2]}}\n```', {"a": {"b": [1, 2]}}),
        (
            'Here you go: {"name": "Ana", "note": "uses {braces}"} Thanks!',
            {"name": "Ana", "note": "uses {braces}"},
        ),
        ('prefix {not json} then {"ok": true}', {"ok": True}),
        ("[1, 2, 3]", None),
        ("no json at all", None),
    ],
)
def test_extract_json_object(text: str, expected: object) -> None:
    assert extract_json_object(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1,234.50 and -3 and 2.5e3 and .5", [1234.5, -3.0, 2500.0, 0.5]),
        ("It is 42.", [42.0]),
        ("2024-05-01", [2024.0, 5.0, 1.0]),
        ("12% of 1,000,000", [12.0, 1_000_000.0]),
        ("version v2.0 and x1", []),
        ("\u22127 degrees", [-7.0]),
    ],
)
def test_extract_numbers(text: str, expected: list[float]) -> None:
    assert extract_numbers(text) == expected
