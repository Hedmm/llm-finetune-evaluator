"""Exception types.

Every error the user can fix (a bad config, a malformed eval set, an unreachable
endpoint) derives from :class:`LfeError`. The CLI prints these as a short message
instead of a traceback.
"""

from __future__ import annotations


class LfeError(Exception):
    """Base class for errors caused by user input or the environment."""


class ConfigError(LfeError):
    """The experiment configuration is invalid."""


class DatasetError(LfeError):
    """An eval set or label file is malformed."""


class BackendError(LfeError):
    """A model backend failed to produce a generation."""


class RetryableBackendError(BackendError):
    """A transient backend failure that is worth retrying.

    Args:
        message: Human-readable description of the failure.
        retry_after: Delay in seconds requested by the server, if any.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ScorerError(LfeError):
    """A scorer cannot score an item, usually because the reference is unusable."""


class RunError(LfeError):
    """Run directories are missing, incomplete or incompatible with each other."""
