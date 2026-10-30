"""Judge calibration: compare the LLM judge with human labels.

An LLM judge is only useful if it agrees with people. Calibration runs the judge
on a small human-labelled sample of (item, output) pairs and reports the raw
agreement and Cohen's kappa, which corrects agreement for chance: a judge that
calls everything "correct" can agree 80% of the time on an easy set and still
have a kappa near zero.

When kappa is below the threshold (0.6 by default, the conventional boundary
between "moderate" and "substantial" agreement) judge-based results are marked
as unreliable in every report.

Human labels are JSONL, one labelled output per line::

    {"id": "faq-017", "output": "Refunds take 5 business days.", "label": "correct"}

``id`` refers to an eval item (for the question and the reference), ``output`` is
the exact text that was labelled and ``label`` is one of the rubric verdicts.
Optional ``model`` and ``annotator`` fields are kept for bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from llm_finetune_evaluator.dataset import EvalItem
from llm_finetune_evaluator.errors import DatasetError
from llm_finetune_evaluator.rubric import JudgeVerdict, normalise_verdict
from llm_finetune_evaluator.scorers.judge import UNPARSEABLE, JudgeScorer
from llm_finetune_evaluator.stats import (
    IntArray,
    bootstrap_ci_statistic,
    cohen_kappa,
    confusion_matrix,
)

DEFAULT_KAPPA_THRESHOLD = 0.6
CALIBRATION_FILE_NAME = "calibration.json"


@dataclass(frozen=True)
class HumanLabel:
    """A human verdict on one model output.

    Attributes:
        item_id: Id of the eval item that was answered.
        output: The exact output text that was labelled.
        label: The human's rubric verdict.
        model: Optional name of the model that produced the output.
        annotator: Optional annotator id.
    """

    item_id: str
    output: str
    label: JudgeVerdict
    model: str | None = None
    annotator: str | None = None


def load_human_labels(path: Path) -> list[HumanLabel]:
    """Load human labels from JSONL.

    Raises:
        DatasetError: If the file is missing or a line is malformed.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise DatasetError(f"human labels file not found: {path}") from error
    labels: list[HumanLabel] = []
    problems: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"line {line_number}: invalid JSON ({error.msg})")
            continue
        if not isinstance(row, dict):
            problems.append(f"line {line_number}: expected a JSON object")
            continue
        missing = [key for key in ("id", "output", "label") if row.get(key) is None]
        if missing:
            problems.append(f"line {line_number}: missing {', '.join(missing)}")
            continue
        verdict = normalise_verdict(str(row["label"]))
        if verdict is None:
            allowed = ", ".join(v.value for v in JudgeVerdict)
            problems.append(f"line {line_number}: label {row['label']!r} is not one of {allowed}")
            continue
        labels.append(
            HumanLabel(
                item_id=str(row["id"]),
                output=str(row["output"]),
                label=verdict,
                model=row.get("model"),
                annotator=row.get("annotator"),
            )
        )
    if problems:
        raise DatasetError(f"{path} has invalid human labels:\n  " + "\n  ".join(problems[:20]))
    if not labels:
        raise DatasetError(f"{path} contains no human labels")
    return labels


@dataclass(frozen=True)
class CalibrationResult:
    """Agreement between the judge and human labels.

    Attributes:
        n: Number of labelled outputs.
        agreement: Share of outputs where judge and human chose the same verdict.
        kappa: Cohen's kappa (NaN when undefined).
        kappa_low: Lower bound of the bootstrap confidence interval for kappa.
        kappa_high: Upper bound of the bootstrap confidence interval for kappa.
        confidence: Coverage of the kappa interval.
        threshold: Minimum kappa for judge results to count as reliable.
        categories: Row and column order of the confusion matrix.
        confusion: Counts with human verdicts as rows and judge verdicts as columns.
        n_unparseable: Judge replies without a recognisable verdict.
        judge: Name of the judge model.
        labels_file: Name of the human labels file.
        labels_sha256: SHA-256 of the human labels file.
        created_at: UTC timestamp.
    """

    n: int
    agreement: float
    kappa: float
    kappa_low: float
    kappa_high: float
    confidence: float
    threshold: float
    categories: tuple[str, ...]
    confusion: tuple[tuple[int, ...], ...]
    n_unparseable: int
    judge: str
    labels_file: str
    labels_sha256: str
    created_at: str

    @property
    def reliable(self) -> bool:
        """Whether kappa reaches the threshold."""
        return not math.isnan(self.kappa) and self.kappa >= self.threshold

    @property
    def warning(self) -> str | None:
        """A warning to show when judge results should not be trusted."""
        if self.reliable:
            return None
        if math.isnan(self.kappa):
            return (
                f"Judge calibration is inconclusive: kappa is undefined because the {self.n} "
                "labelled outputs all share one verdict. Label a more varied sample."
            )
        return (
            f"Judge agreement with human labels is too low: Cohen's kappa = {self.kappa:.2f} "
            f"< {self.threshold:.2f} on {self.n} labelled outputs. "
            "Treat judge-based results as unreliable."
        )

    def per_category(self) -> list[dict[str, Any]]:
        """Per-verdict recall of the judge against the human labels."""
        rows = []
        for index, category in enumerate(self.categories):
            human_total = sum(self.confusion[index])
            if human_total == 0:
                continue
            rows.append(
                {
                    "verdict": category,
                    "human_count": human_total,
                    "judge_agrees": self.confusion[index][index],
                    "recall": self.confusion[index][index] / human_total,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (NaN becomes ``null``)."""
        data = asdict(self)
        for key in ("kappa", "kappa_low", "kappa_high"):
            if isinstance(data[key], float) and math.isnan(data[key]):
                data[key] = None
        data["confusion"] = [list(row) for row in self.confusion]
        data["categories"] = list(self.categories)
        data["reliable"] = self.reliable
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationResult:
        """Rebuild a result from :meth:`to_dict` output."""

        def number(key: str) -> float:
            value = data.get(key)
            return math.nan if value is None else float(value)

        return cls(
            n=int(data["n"]),
            agreement=float(data["agreement"]),
            kappa=number("kappa"),
            kappa_low=number("kappa_low"),
            kappa_high=number("kappa_high"),
            confidence=float(data["confidence"]),
            threshold=float(data["threshold"]),
            categories=tuple(data["categories"]),
            confusion=tuple(tuple(int(x) for x in row) for row in data["confusion"]),
            n_unparseable=int(data["n_unparseable"]),
            judge=str(data["judge"]),
            labels_file=str(data["labels_file"]),
            labels_sha256=str(data["labels_sha256"]),
            created_at=str(data["created_at"]),
        )


def calibrate(
    human: Sequence[JudgeVerdict],
    judge: Sequence[JudgeVerdict | None],
    *,
    threshold: float = DEFAULT_KAPPA_THRESHOLD,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
    judge_name: str = "judge",
    labels_file: Path | None = None,
) -> CalibrationResult:
    """Compute agreement statistics between human and judge verdicts.

    Unparseable judge replies (``None``) form their own category, so they count
    as disagreements.

    Args:
        human: Human verdicts.
        judge: Judge verdicts for the same outputs, aligned with ``human``.
        threshold: Minimum kappa for the judge to count as reliable.
        confidence: Coverage of the bootstrap interval for kappa.
        n_resamples: Bootstrap resamples for the kappa interval.
        seed: Seed for the bootstrap.
        judge_name: Name of the judge model, for the record.
        labels_file: Path of the human labels file, for the record.
    """
    if len(human) != len(judge):
        raise ValueError("human and judge verdicts must be aligned")
    if not human:
        raise ValueError("at least one labelled output is required")
    human_labels = [verdict.value for verdict in human]
    judge_labels = [verdict.value if verdict is not None else UNPARSEABLE for verdict in judge]
    categories = [verdict.value for verdict in JudgeVerdict]
    if UNPARSEABLE in judge_labels:
        categories.append(UNPARSEABLE)
    matrix = confusion_matrix(human_labels, judge_labels, categories)
    kappa = cohen_kappa(human_labels, judge_labels, categories)
    human_array = np.array(human_labels)
    judge_array = np.array(judge_labels)

    def resampled_kappa(indices: IntArray) -> float:
        return cohen_kappa(list(human_array[indices]), list(judge_array[indices]), categories)

    interval = bootstrap_ci_statistic(
        len(human_labels),
        resampled_kappa,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    sha = hashlib.sha256(labels_file.read_bytes()).hexdigest() if labels_file else ""
    return CalibrationResult(
        n=len(human_labels),
        agreement=float(np.trace(matrix) / len(human_labels)),
        kappa=kappa,
        kappa_low=interval.low,
        kappa_high=interval.high,
        confidence=confidence,
        threshold=threshold,
        categories=tuple(categories),
        confusion=tuple(tuple(int(x) for x in row) for row in matrix.tolist()),
        n_unparseable=judge_labels.count(UNPARSEABLE),
        judge=judge_name,
        labels_file=labels_file.name if labels_file else "",
        labels_sha256=sha,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


async def calibrate_judge(
    scorer: JudgeScorer,
    labels: Sequence[HumanLabel],
    items: Mapping[str, EvalItem],
    *,
    threshold: float = DEFAULT_KAPPA_THRESHOLD,
    seed: int = 0,
    labels_file: Path | None = None,
) -> CalibrationResult:
    """Run the judge on human-labelled outputs and compare the verdicts.

    Args:
        scorer: The judge scorer (its engine caches the judge's replies).
        labels: Human-labelled outputs.
        items: Eval items by id, for the questions and references.
        threshold: Minimum kappa for the judge to count as reliable.
        seed: Seed for the kappa interval.
        labels_file: Path of the labels file, for the record.

    Raises:
        DatasetError: If a label refers to an item that is not in the eval sets.
    """
    unknown = sorted({label.item_id for label in labels} - set(items))
    if unknown:
        shown = ", ".join(unknown[:10])
        raise DatasetError(f"human labels refer to items that are not in the eval sets: {shown}")
    verdicts = await scorer.verdicts(
        [items[label.item_id] for label in labels], [label.output for label in labels]
    )
    return calibrate(
        [label.label for label in labels],
        [verdict for verdict, _, _ in verdicts],
        threshold=threshold,
        seed=seed,
        judge_name=scorer.engine.backend.name,
        labels_file=labels_file,
    )


def save_calibration(result: CalibrationResult, path: Path) -> None:
    """Write a calibration result as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_calibration(path: Path) -> CalibrationResult | None:
    """Read a calibration result, or return ``None`` if the file does not exist."""
    if not path.exists():
        return None
    return CalibrationResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


def labeling_template(
    outputs: Sequence[tuple[EvalItem, str]], size: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Draw a random sample of outputs for humans to label.

    Identical (item, output) pairs from different models are labelled once,
    and the model name is left out so that annotators grade blind, like the
    judge.

    Args:
        outputs: ``(item, output)`` pairs from one or more runs.
        size: Number of outputs to sample (all of them if there are fewer).
        seed: Seed for the sample.

    Returns:
        Rows with ``id``, ``question``, ``reference``, ``output`` and an empty
        ``label`` for the annotator to fill in.
    """
    unique = list({(item.id, output): (item, output) for item, output in outputs}.values())
    chosen = random.Random(seed).sample(unique, min(size, len(unique)))
    return [
        {
            "id": item.id,
            "question": item.input_text(),
            "reference": item.reference,
            "output": output,
            "label": None,
        }
        for item, output in chosen
    ]
