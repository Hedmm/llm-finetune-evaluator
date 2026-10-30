"""Run directories: how results are written to and read from disk.

Layout of one experiment::

    runs/<experiment>/
        experiment.json      config snapshot, baseline and model order
        calibration.json     judge calibration (optional)
        <model>/
            run.json         metadata: model, sampling, eval set hashes, metrics
            records.jsonl    one line per item: prompt, output and every score
        report.md, report.html, analysis.json   written by ``lfe report``

Each model's run is self-contained, so ``lfe compare`` can pair any two runs
that were scored on the same eval set content.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_finetune_evaluator.dataset import ChatMessage, TaskType
from llm_finetune_evaluator.errors import RunError
from llm_finetune_evaluator.scorers.base import MetricInfo, Score

RUN_FORMAT_VERSION = 1
RUN_FILE = "run.json"
RECORDS_FILE = "records.jsonl"
EXPERIMENT_FILE = "experiment.json"


def display_path(path: Path) -> str:
    """Return ``path`` relative to the working directory when possible."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass(frozen=True)
class ItemRecord:
    """Everything recorded for one item of one model's run.

    Attributes:
        id: Item id.
        eval_set: Name of the eval set the item belongs to.
        task: Task type.
        tags: Slice tags.
        input: The item input as plain text (for reports).
        messages: The exact messages sent to the model.
        reference: The reference answer.
        output: The output that was scored (reasoning blocks removed).
        raw_output: The raw model output when it differs from ``output``.
        finish_reason: Why generation stopped, if reported.
        latency_s: Generation latency in seconds, if known.
        cached: Whether the generation came from the cache.
        scores: Score of every metric that applies to the item.
    """

    id: str
    eval_set: str
    task: TaskType
    tags: dict[str, str]
    input: str
    messages: tuple[ChatMessage, ...]
    reference: Any
    output: str
    raw_output: str | None = None
    finish_reason: str | None = None
    latency_s: float | None = None
    cached: bool = False
    scores: dict[str, Score] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        data: dict[str, Any] = {
            "id": self.id,
            "eval_set": self.eval_set,
            "task": self.task.value,
            "tags": self.tags,
            "input": self.input,
            "messages": [message.model_dump() for message in self.messages],
            "reference": self.reference,
            "output": self.output,
        }
        if self.raw_output is not None:
            data["raw_output"] = self.raw_output
        data["finish_reason"] = self.finish_reason
        data["latency_s"] = self.latency_s
        data["cached"] = self.cached
        data["scores"] = {name: score.to_dict() for name, score in self.scores.items()}
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ItemRecord:
        """Rebuild a record from :meth:`to_dict` output."""
        return cls(
            id=str(data["id"]),
            eval_set=str(data["eval_set"]),
            task=TaskType(data["task"]),
            tags={str(k): str(v) for k, v in (data.get("tags") or {}).items()},
            input=str(data.get("input", "")),
            messages=tuple(ChatMessage.model_validate(m) for m in data.get("messages", [])),
            reference=data.get("reference"),
            output=str(data["output"]),
            raw_output=data.get("raw_output"),
            finish_reason=data.get("finish_reason"),
            latency_s=data.get("latency_s"),
            cached=bool(data.get("cached", False)),
            scores={
                name: Score.from_dict(score) for name, score in (data.get("scores") or {}).items()
            },
        )


@dataclass(frozen=True)
class RunData:
    """One model's run loaded from disk.

    Attributes:
        path: The run directory.
        meta: Contents of ``run.json``.
        records: Item records in eval set order.
    """

    path: Path
    meta: dict[str, Any]
    records: tuple[ItemRecord, ...]

    @property
    def model(self) -> str:
        """Model name."""
        return str(self.meta["model"])

    @property
    def metrics(self) -> list[MetricInfo]:
        """Metrics scored in this run, in config order."""
        return [MetricInfo.from_dict(metric) for metric in self.meta.get("metrics", [])]

    @property
    def eval_set_hash(self) -> str:
        """Combined content hash of the eval sets."""
        return str(self.meta["eval_set_hash"])

    def by_id(self) -> dict[str, ItemRecord]:
        """Records keyed by item id."""
        return {record.id: record for record in self.records}


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to a temporary sibling file, then rename it over ``path``.

    Readers never see a half-written file, and the file gets the usual
    permissions (the process umask applies, unlike with ``tempfile.mkstemp``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` as indented JSON, atomically."""
    _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_run(directory: Path, meta: Mapping[str, Any], records: Iterable[ItemRecord]) -> None:
    """Write a model's run directory, replacing any previous run there."""
    lines = "".join(json.dumps(record.to_dict(), ensure_ascii=False) + "\n" for record in records)
    _atomic_write(directory / RECORDS_FILE, lines)
    write_json(directory / RUN_FILE, {"format_version": RUN_FORMAT_VERSION, **meta})


def load_run(directory: Path) -> RunData:
    """Load a model's run directory.

    Raises:
        RunError: If the directory is not a complete run.
    """
    run_file = directory / RUN_FILE
    records_file = directory / RECORDS_FILE
    if not run_file.is_file() or not records_file.is_file():
        raise RunError(
            f"{directory} is not a run directory (expected {RUN_FILE} and {RECORDS_FILE})"
        )
    try:
        meta = json.loads(run_file.read_text(encoding="utf-8"))
        records = tuple(
            ItemRecord.from_dict(json.loads(line))
            for line in records_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (json.JSONDecodeError, KeyError, ValueError) as error:
        raise RunError(f"{directory} contains a corrupt run: {error}") from error
    version = meta.get("format_version")
    if version != RUN_FORMAT_VERSION:
        raise RunError(
            f"{directory} uses run format {version}; this version reads {RUN_FORMAT_VERSION}"
        )
    return RunData(path=directory, meta=meta, records=records)


def load_experiment(directory: Path) -> dict[str, Any] | None:
    """Read ``experiment.json`` from ``directory``, or ``None`` if there is none."""
    path = directory / EXPERIMENT_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def discover_runs(root: Path) -> list[Path]:
    """Find run directories at ``root`` or up to two levels below it.

    When ``root`` holds an ``experiment.json``, its model list decides which
    runs belong to the experiment and in which order; otherwise runs are
    returned in path order.

    Raises:
        RunError: If no run directory is found.
    """
    if not root.exists():
        raise RunError(f"{root} does not exist")
    if (root / RUN_FILE).is_file():
        return [root]
    experiment = load_experiment(root)
    if experiment is not None:
        ordered = [root / name for name in experiment.get("models", [])]
        found = [path for path in ordered if (path / RUN_FILE).is_file()]
        if found:
            return found
    candidates = sorted(
        {path.parent for pattern in ("*/run.json", "*/*/run.json") for path in root.glob(pattern)}
    )
    if not candidates:
        raise RunError(f"no runs found in {root}; run `lfe run <config>` first")
    return candidates
