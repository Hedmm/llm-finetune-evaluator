"""Backend that replays recorded predictions from a JSONL file.

Use it to evaluate outputs produced elsewhere, for example by a training script
or a batch inference job: each line holds an item id and the model output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_finetune_evaluator.backends.base import Backend, Generation, GenerationRequest
from llm_finetune_evaluator.errors import BackendError, DatasetError


def load_predictions(
    path: Path, id_field: str = "id", output_field: str = "output"
) -> dict[str, str]:
    """Read ``{id: output}`` from a JSONL file of predictions.

    Raises:
        DatasetError: If the file is missing, a line is malformed, or an id repeats.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise DatasetError(f"predictions file not found: {path}") from error
    predictions: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetError(f"{path}, line {line_number}: invalid JSON ({error.msg})") from error
        if not isinstance(row, dict) or id_field not in row or output_field not in row:
            raise DatasetError(
                f"{path}, line {line_number}: expected an object with "
                f"'{id_field}' and '{output_field}'"
            )
        item_id = str(row[id_field])
        if item_id in predictions:
            raise DatasetError(f"{path}, line {line_number}: duplicate id {item_id!r}")
        output = row[output_field]
        predictions[item_id] = output if isinstance(output, str) else json.dumps(output)
    return predictions


class ReplayBackend(Backend):
    """Returns the recorded output for each item id.

    Args:
        name: Display name in reports.
        path: JSONL file with one prediction per line.
        id_field: Field that holds the item id.
        output_field: Field that holds the model output.
    """

    kind = "replay"

    def __init__(
        self, name: str, path: Path, *, id_field: str = "id", output_field: str = "output"
    ) -> None:
        super().__init__(name, max_concurrency=1, max_batch_size=256, cacheable=False)
        self.path = path
        self.predictions = load_predictions(path, id_field, output_field)
        self._sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def identity(self) -> dict[str, Any]:
        """Return the file name and its SHA-256 hash."""
        return {"backend": self.kind, "file": self.path.name, "sha256": self._sha256}

    def request_identity(self, request: GenerationRequest) -> dict[str, Any]:
        """Replayed outputs depend on the item id."""
        return {"item_id": request.item_id}

    async def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        """Look up each request's item id."""
        generations = []
        for request in requests:
            output = self.predictions.get(request.item_id or "")
            if output is None:
                raise BackendError(
                    f"{self.name}: {self.path} has no prediction for item {request.item_id!r}"
                )
            generations.append(Generation(text=output))
        return generations
