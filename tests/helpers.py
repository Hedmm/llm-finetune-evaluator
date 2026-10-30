"""Helpers shared by several test modules."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from llm_finetune_evaluator.dataset import ChatMessage, TaskType
from llm_finetune_evaluator.scorers.base import MetricInfo, Score
from llm_finetune_evaluator.storage import ItemRecord, RunData, load_run, write_run


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write ``rows`` as JSON lines and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON lines file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_yaml(path: Path, data: Mapping[str, Any]) -> Path:
    """Write ``data`` as YAML and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(data), sort_keys=False), encoding="utf-8")
    return path


def qa_rows(n: int, topics: Sequence[str] = ("billing", "shipping")) -> list[dict[str, Any]]:
    """Open QA items with a reference answer and a topic tag."""
    return [
        {
            "id": f"q{i:03d}",
            "input": f"Question {i}?",
            "reference": f"answer {i}",
            "task": "open_qa",
            "tags": {"topic": topics[i % len(topics)]},
        }
        for i in range(n)
    ]


def mock_experiment(
    tmp_path: Path,
    *,
    n: int = 120,
    models: Sequence[Mapping[str, Any]] | None = None,
    scorers: Sequence[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Write a small mock experiment (eval set and config) and return the config path."""
    write_jsonl(tmp_path / "data" / "qa.jsonl", rows if rows is not None else qa_rows(n))
    config: dict[str, Any] = {
        "name": "demo",
        "output_dir": "runs",
        "eval_sets": [{"name": "qa", "path": "data/qa.jsonl"}],
        "models": list(
            models
            or [
                {"name": "base", "backend": "mock", "accuracy": 0.6},
                {"name": "tuned", "backend": "mock", "accuracy": 0.85},
            ]
        ),
        "scorers": list(scorers or [{"type": "normalized_match"}, {"type": "refusal"}]),
        "stats": {"bootstrap_resamples": 500, "permutation_resamples": 500},
    }
    config.update(extra or {})
    return write_yaml(tmp_path / "config.yaml", config)


BINARY = MetricInfo(
    name="accuracy",
    scorer="exact_match",
    description="Exact match",
    higher_is_better=True,
    binary=True,
    tasks=("open_qa",),
)


def make_run(
    directory: Path,
    model: str,
    values: Mapping[str, float | None],
    *,
    tags: Mapping[str, Mapping[str, str]] | None = None,
    metrics: Sequence[MetricInfo] = (BINARY,),
    labels: Mapping[str, str] | None = None,
    eval_set_hash: str = "hash-1",
    is_baseline: bool = False,
    model_index: int = 0,
    details: Mapping[str, Mapping[str, Any]] | None = None,
) -> RunData:
    """Write and load a run whose every metric has the given per-item values."""
    records = [
        ItemRecord(
            id=item_id,
            eval_set="qa",
            task=TaskType.OPEN_QA,
            tags=dict((tags or {}).get(item_id, {})),
            input=f"Input of {item_id}",
            messages=(ChatMessage(role="user", content=f"Input of {item_id}"),),
            reference=f"reference {item_id}",
            output=f"{model} output for {item_id}",
            scores={
                metric.name: Score(
                    value,
                    label=(labels or {}).get(item_id),
                    details=dict((details or {}).get(item_id, {})),
                )
                for metric in metrics
            },
        )
        for item_id, value in values.items()
    ]
    meta = {
        "experiment": "unit",
        "model": model,
        "is_baseline": is_baseline,
        "model_index": model_index,
        "backend": {"name": model, "kind": "mock"},
        "sampling": {"temperature": 0.0},
        "metrics": [metric.to_dict() for metric in metrics],
        "eval_sets": [
            {"name": "qa", "path": "qa.jsonl", "sha256": eval_set_hash, "n_items": len(records)}
        ],
        "eval_set_hash": eval_set_hash,
        "generation": {
            "requested": len(records),
            "cached": 0,
            "generated": len(records),
            "retries": 0,
        },
    }
    write_run(directory / model, meta, records)
    return load_run(directory / model)
