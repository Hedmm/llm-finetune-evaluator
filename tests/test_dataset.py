from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from helpers import write_jsonl
from llm_finetune_evaluator.dataset import (
    ChatMessage,
    EvalItem,
    PromptTemplate,
    TaskType,
    load_eval_set,
    render_messages,
)
from llm_finetune_evaluator.errors import DatasetError

ROWS: list[dict[str, Any]] = [
    {
        "id": "c1",
        "input": "My card has not arrived.",
        "reference": "card_arrival",
        "task": "classification",
        "tags": {"group": "card", "length": 5},
    },
    {
        "id": "e1",
        "input": [
            {"role": "system", "content": "Extract the fields."},
            {"role": "user", "content": "Order 42 for Ana."},
        ],
        "reference": {"order_id": 42, "name": "Ana"},
        "task": "extraction",
    },
    {"id": "q1", "input": "Capital of France?", "reference": ["Paris"], "task": "open_qa"},
    {"id": "t1", "input": "What is the CEO's salary?", "reference": None, "task": "trap"},
]


def test_loads_items_in_file_order(tmp_path: Path) -> None:
    eval_set = load_eval_set(write_jsonl(tmp_path / "set.jsonl", ROWS))
    assert eval_set.name == "set"
    assert [item.id for item in eval_set.items] == ["c1", "e1", "q1", "t1"]
    assert eval_set.items[0].tags == {"group": "card", "length": "5"}
    assert eval_set.items[1].input[0] == ChatMessage(role="system", content="Extract the fields.")
    assert eval_set.items[3].task is TaskType.TRAP
    assert eval_set.items[3].task.expects_abstention


def test_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text("\n" + json.dumps(ROWS[0]) + "\n\n", encoding="utf-8")
    assert len(load_eval_set(path).items) == 1


def test_reports_every_problem_with_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    lines = [
        json.dumps(ROWS[0]),
        "{not json",
        json.dumps({"id": "x", "input": "hi", "reference": "a"}),
        json.dumps({"id": "y", "input": "hi", "reference": "a", "task": "summarise"}),
        json.dumps({"id": "z", "input": "hi", "reference": "not a dict", "task": "extraction"}),
        json.dumps(ROWS[0]),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(DatasetError) as caught:
        load_eval_set(path)
    message = str(caught.value)
    assert "line 2: invalid JSON" in message
    assert "line 3, field 'task'" in message
    assert "line 4, field 'task'" in message
    assert "line 5" in message
    assert "JSON object as reference" in message
    assert "line 6: duplicate id 'c1' (first seen on line 1)" in message


def test_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no items"):
        load_eval_set(path)


def test_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_eval_set(tmp_path / "missing.jsonl")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"id": " ", "input": "x", "reference": "a", "task": "open_qa"}, "blank"),
        ({"id": "a", "input": "  ", "reference": "a", "task": "open_qa"}, "blank"),
        (
            {"id": "a", "input": [{"role": "system", "content": "x"}], "task": "trap"},
            "user message",
        ),
        ({"id": "a", "input": "x", "reference": [], "task": "open_qa"}, "at least one"),
        ({"id": "a", "input": "x", "reference": 3, "task": "classification"}, "strings"),
        ({"id": "a", "input": "x", "reference": {"a": 1}, "task": "trap"}, "null or a note"),
        (
            {"id": "a", "input": "x", "reference": "a", "task": "open_qa", "tags": {"task": "x"}},
            "reserved",
        ),
        (
            {"id": "a", "input": "x", "reference": "a", "task": "open_qa", "tags": {"t": [1]}},
            "string",
        ),
        (
            {"id": "a", "input": "x", "reference": "a", "task": "open_qa", "extra": 1},
            "Extra inputs",
        ),
    ],
)
def test_validates_items(row: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EvalItem.model_validate(row)


def test_numbers_and_booleans_are_accepted_as_tags() -> None:
    item = EvalItem.model_validate(
        {"id": "a", "input": "x", "reference": 4, "task": "open_qa", "tags": {"n": 2, "ok": True}}
    )
    assert item.tags == {"n": "2", "ok": "true"}
    assert item.references() == [4]


def test_content_hash_ignores_formatting_but_not_content(tmp_path: Path) -> None:
    compact = write_jsonl(tmp_path / "a.jsonl", ROWS)
    spaced = tmp_path / "b.jsonl"
    spaced.write_text(
        "\n\n".join(json.dumps(dict(reversed(list(row.items())))) for row in ROWS),
        encoding="utf-8",
    )
    changed_rows = [dict(ROWS[0], reference="card_linking"), *ROWS[1:]]
    changed = write_jsonl(tmp_path / "c.jsonl", changed_rows)
    assert load_eval_set(compact).content_hash == load_eval_set(spaced).content_hash
    assert load_eval_set(compact).content_hash != load_eval_set(changed).content_hash


def test_input_text_joins_chat_messages() -> None:
    item = EvalItem.model_validate(ROWS[1])
    assert item.input_text() == "[system] Extract the fields.\n\n[user] Order 42 for Ana."


class TestRenderMessages:
    def test_plain_input_becomes_one_user_message(self) -> None:
        item = EvalItem.model_validate(ROWS[2])
        assert render_messages(item) == (ChatMessage(role="user", content="Capital of France?"),)

    def test_applies_template_system_prompt_and_variables(self) -> None:
        prompt = PromptTemplate(
            system="You classify messages for {{ company }}.",
            template=(
                "Labels: {{ labels | join(', ') }}\nGroup: {{ tags.group }}\nMessage: {{ input }}"
            ),
            variables={"company": "a bank", "labels": ["card_arrival", "card_linking"]},
        )
        messages = render_messages(EvalItem.model_validate(ROWS[0]), prompt)
        assert messages[0] == ChatMessage(
            role="system", content="You classify messages for a bank."
        )
        assert messages[1].content == (
            "Labels: card_arrival, card_linking\nGroup: card\nMessage: My card has not arrived."
        )

    def test_json_braces_need_no_escaping(self) -> None:
        prompt = PromptTemplate(template='Reply like {"intent": "..."}.\n{{ input }}')
        messages = render_messages(EvalItem.model_validate(ROWS[0]), prompt)
        assert messages[0].content.startswith('Reply like {"intent": "..."}.')

    def test_chat_input_keeps_its_own_system_message(self) -> None:
        prompt = PromptTemplate(system="Ignored because the chat has a system message.")
        messages = render_messages(EvalItem.model_validate(ROWS[1]), prompt)
        assert [m.content for m in messages] == ["Extract the fields.", "Order 42 for Ana."]

    def test_system_prompt_is_added_to_chat_input_without_one(self) -> None:
        item = EvalItem.model_validate(
            {"id": "a", "input": [{"role": "user", "content": "Hi"}], "task": "trap"}
        )
        messages = render_messages(item, PromptTemplate(system="Be brief."))
        assert [m.role for m in messages] == ["system", "user"]

    def test_unknown_names_are_reported_with_the_available_ones(self) -> None:
        prompt = PromptTemplate(template="{{ question }}")
        with pytest.raises(
            DatasetError, match=r"'question' is undefined\. Available names: id, input"
        ):
            render_messages(EvalItem.model_validate(ROWS[2]), prompt)

    def test_reference_is_not_available_to_templates(self) -> None:
        prompt = PromptTemplate(template="{{ reference }}")
        with pytest.raises(DatasetError, match="'reference' is undefined"):
            render_messages(EvalItem.model_validate(ROWS[2]), prompt)

    def test_syntax_errors_are_reported(self) -> None:
        with pytest.raises(DatasetError, match="syntax error"):
            render_messages(EvalItem.model_validate(ROWS[2]), PromptTemplate(template="{{ input "))

    def test_variables_cannot_shadow_built_in_names(self) -> None:
        with pytest.raises(ValueError, match="clash"):
            PromptTemplate(variables={"input": "x"})
