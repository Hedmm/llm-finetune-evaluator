"""On-disk cache of generations.

Generations are stored in a SQLite file, keyed by a SHA-256 hash of the model
identity, the prompt and the sampling parameters. Re-running an experiment, or
re-scoring it with different scorers, reuses every generation that is already
in the cache, so it costs no API calls or GPU time.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from llm_finetune_evaluator.backends.base import Generation
from llm_finetune_evaluator.dataset import canonical_json

CACHE_FILE_NAME = "generations.sqlite"
_KEY_VERSION = 1


def default_cache_dir() -> Path:
    """Return the cache directory.

    ``$LFE_CACHE_DIR`` if set, else ``$XDG_CACHE_HOME/llm-finetune-evaluator``,
    else ``~/.cache/llm-finetune-evaluator``.
    """
    explicit = os.environ.get("LFE_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "llm-finetune-evaluator"


def cache_key(
    identity: Mapping[str, Any], request_identity: Mapping[str, Any], payload: Mapping[str, Any]
) -> str:
    """Return the cache key for a request.

    Args:
        identity: The backend identity (model, endpoint, adapter, ...).
        request_identity: Request fields beyond the prompt that affect output.
        payload: The prompt messages and sampling parameters.
    """
    material = {
        "version": _KEY_VERSION,
        "model": identity,
        "request": request_identity,
        "input": payload,
    }
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


class GenerationCache:
    """A SQLite-backed key-value store of generations.

    The cache is safe to share between concurrent requests of one process and,
    thanks to SQLite's write-ahead log, between processes.

    Args:
        path: The SQLite file; parent directories are created.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS generations ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        self._connection.commit()
        self.hits = 0
        self.misses = 0

    @classmethod
    def open_default(cls, directory: Path | None = None) -> GenerationCache:
        """Open the cache file in ``directory`` or in :func:`default_cache_dir`."""
        return cls((directory or default_cache_dir()) / CACHE_FILE_NAME)

    def get(self, key: str) -> Generation | None:
        """Return the cached generation for ``key``, or ``None``."""
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM generations WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return Generation.from_dict(json.loads(row[0]), cached=True)

    def put(self, key: str, generation: Generation) -> None:
        """Store ``generation`` under ``key``, replacing any previous entry."""
        value = json.dumps(generation.to_dict(), ensure_ascii=False)
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO generations (key, value, created_at) VALUES (?, ?, ?)",
                (key, value, created_at),
            )
            self._connection.commit()

    def __len__(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) FROM generations").fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> GenerationCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
