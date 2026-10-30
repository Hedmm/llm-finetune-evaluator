"""Eval set format: loading, validation, hashing and prompt rendering.

An eval set is a JSONL file with one item per line::

    {
        "id": "faq-001",
        "input": "How long do refunds take?",
        "reference": "5 business days",
        "task": "open_qa",
        "tags": {"topic": "refunds", "difficulty": "easy"},
    }

Fields:

* ``id``: unique, non-empty string.
* ``input``: a prompt string, or a list of chat messages
  (``{"role": "system" | "user" | "assistant", "content": "..."}``).
* ``reference``: the expected answer. A string, a list of acceptable answers
  (or of labels, for multi-label classification), a number, a JSON object for
  extraction, or ``null`` for trap and refusal_expected items.
* ``task``: ``classification``, ``extraction``, ``open_qa``, ``trap`` or
  ``refusal_expected``.
* ``tags``: optional mapping from a slice dimension to a value, for example
  ``{"topic": "billing", "difficulty": "hard", "source": "faq"}``. Every
  metric is reported per tag value.
* ``metadata``: optional free-form object; it is kept but never scored.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import jinja2
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from llm_finetune_evaluator.errors import DatasetError

RESERVED_TAG_KEYS = frozenset({"task", "eval_set"})
"""Tag keys that would clash with the automatic ``task`` and ``eval_set`` slices."""

_MAX_REPORTED_ERRORS = 20


class TaskType(str, Enum):
    """Kind of item, which decides the default scorers and what counts as correct."""

    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    OPEN_QA = "open_qa"
    TRAP = "trap"
    REFUSAL_EXPECTED = "refusal_expected"

    @property
    def expects_abstention(self) -> bool:
        """Whether the correct behaviour is to decline or to say the answer is unknown."""
        return self in ABSTENTION_TASKS


ANSWERABLE_TASKS = frozenset({TaskType.CLASSIFICATION, TaskType.EXTRACTION, TaskType.OPEN_QA})
"""Tasks with a reference answer; a refusal on these items is a needless refusal."""

ABSTENTION_TASKS = frozenset({TaskType.TRAP, TaskType.REFUSAL_EXPECTED})
"""Tasks where the right response is to decline or to say the answer is not available."""


class ChatMessage(BaseModel):
    """One message of a chat conversation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


def canonical_json(value: Any) -> str:
    """Serialise ``value`` deterministically (sorted keys, no whitespace)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class EvalItem(BaseModel):
    """One item of an eval set. See the module docstring for the field semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    input: str | tuple[ChatMessage, ...]
    reference: Any = None
    task: TaskType
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _strip_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("id must not be blank")
        return stripped

    @field_validator("input")
    @classmethod
    def _check_input(cls, value: str | tuple[ChatMessage, ...]) -> str | tuple[ChatMessage, ...]:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("input must not be blank")
        elif not any(message.role == "user" for message in value):
            raise ValueError("chat input must contain at least one user message")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        coerced: dict[str, str] = {}
        for key, tag in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("tag names must be non-empty strings")
            if key in RESERVED_TAG_KEYS:
                raise ValueError(f"tag name {key!r} is reserved for automatic slices")
            if isinstance(tag, bool):
                coerced[key] = "true" if tag else "false"
            elif isinstance(tag, (str, int, float)):
                coerced[key] = str(tag)
            else:
                raise ValueError(f"tag {key!r} must be a string or a number")
        return coerced

    @model_validator(mode="after")
    def _check_reference(self) -> EvalItem:
        reference = self.reference
        if self.task is TaskType.EXTRACTION:
            if not isinstance(reference, dict):
                raise ValueError("extraction items need a JSON object as reference")
        elif self.task in (TaskType.CLASSIFICATION, TaskType.OPEN_QA):
            answers = reference if isinstance(reference, list) else [reference]
            if not answers:
                raise ValueError(f"{self.task.value} items need at least one reference answer")
            for answer in answers:
                is_number = isinstance(answer, (int, float)) and not isinstance(answer, bool)
                if not (isinstance(answer, str) and answer.strip()) and not is_number:
                    raise ValueError(
                        f"{self.task.value} references must be non-empty strings or numbers"
                    )
                if is_number and self.task is TaskType.CLASSIFICATION:
                    raise ValueError("classification labels must be strings")
        elif reference is not None and not isinstance(reference, str):
            raise ValueError(f"{self.task.value} items take null or a note string as reference")
        return self

    def references(self) -> list[Any]:
        """Return the acceptable reference answers as a list (empty for null)."""
        if self.reference is None:
            return []
        if isinstance(self.reference, list):
            return list(self.reference)
        return [self.reference]

    def input_text(self) -> str:
        """Return the input as plain text, joining chat messages with their roles."""
        if isinstance(self.input, str):
            return self.input
        return "\n\n".join(f"[{message.role}] {message.content}" for message in self.input)


@dataclass(frozen=True)
class EvalSet:
    """A loaded and validated eval set.

    Attributes:
        name: Name used in reports and as the ``eval_set`` slice value.
        path: File the items were read from.
        items: Items in file order.
        content_hash: SHA-256 of the canonical JSON of all items. Two files with
            the same items have the same hash, whatever their formatting.
    """

    name: str
    path: Path
    items: tuple[EvalItem, ...]
    content_hash: str


def hash_items(items: Iterable[EvalItem]) -> str:
    """Return the SHA-256 content hash of ``items`` in order."""
    digest = hashlib.sha256()
    for item in items:
        digest.update(canonical_json(item.model_dump(mode="json")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _format_validation_error(line_number: int, error: ValidationError) -> list[str]:
    messages = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"])
        prefix = f"line {line_number}" + (f", field '{location}'" if location else "")
        messages.append(f"{prefix}: {detail['msg']}")
    return messages


def parse_eval_items(lines: Iterable[str], source: str) -> list[EvalItem]:
    """Parse and validate JSONL lines into items.

    Args:
        lines: Lines of a JSONL file. Blank lines are skipped.
        source: Name of the source, used in error messages.

    Raises:
        DatasetError: If any line is not valid JSON, fails validation, or
            repeats an earlier id. All problems (up to 20) are reported at once.
    """
    items: list[EvalItem] = []
    problems: list[str] = []
    seen: dict[str, int] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"line {line_number}: invalid JSON ({error.msg})")
            continue
        try:
            item = EvalItem.model_validate(raw)
        except ValidationError as error:
            problems.extend(_format_validation_error(line_number, error))
            continue
        if item.id in seen:
            problems.append(
                f"line {line_number}: duplicate id {item.id!r} (first seen on line {seen[item.id]})"
            )
            continue
        seen[item.id] = line_number
        items.append(item)
    if problems:
        shown = problems[:_MAX_REPORTED_ERRORS]
        more = len(problems) - len(shown)
        suffix = f"\n  ... and {more} more" if more else ""
        raise DatasetError(f"{source} is not a valid eval set:\n  " + "\n  ".join(shown) + suffix)
    if not items:
        raise DatasetError(f"{source} contains no items")
    return items


def load_eval_set(path: Path, name: str | None = None) -> EvalSet:
    """Load and validate an eval set from a JSONL file.

    Args:
        path: Path to the JSONL file.
        name: Name for reports; defaults to the file stem.

    Raises:
        DatasetError: If the file is missing or invalid.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise DatasetError(f"eval set not found: {path}") from error
    items = parse_eval_items(text.splitlines(), str(path))
    return EvalSet(
        name=name or path.stem, path=path, items=tuple(items), content_hash=hash_items(items)
    )


_BUILTIN_TEMPLATE_NAMES = frozenset({"input", "id", "task", "tags", "metadata"})


class PromptTemplate(BaseModel):
    """How an item's ``input`` becomes the messages sent to a model.

    ``system`` and ``template`` are Jinja2 templates. They can use ``input``,
    ``id``, ``task``, ``tags`` (``{{ tags.topic }}``), ``metadata`` and every
    key of ``variables``; an unknown name is an error. Lists can be joined with
    ``{{ labels | join(", ") }}``. The reference answer is never available, so
    a template cannot leak it into the prompt.

    ``template`` applies to string inputs only; chat inputs are sent as they
    are. ``system`` is added as the first message unless the conversation
    already starts with a system message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str | None = None
    template: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)

    @field_validator("variables")
    @classmethod
    def _check_variables(cls, value: dict[str, Any]) -> dict[str, Any]:
        clashes = sorted(set(value) & _BUILTIN_TEMPLATE_NAMES)
        if clashes:
            raise ValueError(f"prompt variables {clashes} clash with built-in names")
        return value


_TEMPLATE_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined, autoescape=False, keep_trailing_newline=True
)


def _render(template: str, context: Mapping[str, Any], item_id: str) -> str:
    try:
        return _TEMPLATE_ENV.from_string(template).render(**context)
    except jinja2.UndefinedError as error:
        available = ", ".join(sorted(context))
        raise DatasetError(
            f"prompt template for item {item_id!r}: {error.message}. Available names: {available}"
        ) from error
    except jinja2.TemplateSyntaxError as error:
        raise DatasetError(f"prompt template syntax error: {error.message}") from error


def render_messages(
    item: EvalItem, prompt: PromptTemplate | None = None
) -> tuple[ChatMessage, ...]:
    """Build the chat messages sent to a model for ``item``.

    Args:
        item: The eval item.
        prompt: Optional system prompt and template; ``None`` sends the input as
            a single user message (or the chat input unchanged).

    Raises:
        DatasetError: If a template uses an unknown name or has a syntax error.
    """
    prompt = prompt or PromptTemplate()
    context: dict[str, Any] = {
        **prompt.variables,
        "id": item.id,
        "task": item.task.value,
        "tags": item.tags,
        "metadata": item.metadata,
    }
    if isinstance(item.input, str):
        context["input"] = item.input
        content = _render(prompt.template, context, item.id) if prompt.template else item.input
        messages: list[ChatMessage] = [ChatMessage(role="user", content=content)]
    else:
        context["input"] = item.input_text()
        messages = list(item.input)
    if prompt.system and messages[0].role != "system":
        system = _render(prompt.system, context, item.id)
        messages.insert(0, ChatMessage(role="system", content=system))
    return tuple(messages)
