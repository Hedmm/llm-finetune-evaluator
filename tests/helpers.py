"""Helpers shared by several test modules."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write ``rows`` as JSON lines and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON lines file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
